#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""

import logging
import os
import sys
from dataclasses import dataclass

import torch
import torch_npu
import torch_sip

# 配置全局日志 (G.LOG.02)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("torch_sip_strmm_test")


@dataclass(frozen=True)
class TrmmConfig:
    """参数对象：封装 STRMM 的所有维度、标量及模式配置"""
    m: int
    n: int
    alpha: float
    side: str = "L"
    uplo: str = "L"
    trans: str = "N"
    name: str = "Case"


class StrmmTester:
    """单精度三角矩阵乘法 (STRMM) 算子测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        blocking_env = os.getenv("ASCEND_LAUNCH_BLOCKING", "0")
        mode_str = "异步 (Async)" if blocking_env == "0" else "同步 (Blocking)"
        logger.info("当前 NPU 运行模式: %s (ASCEND_LAUNCH_BLOCKING=%s)", mode_str, blocking_env)

    @staticmethod
    def to_column_major(tensor: torch.Tensor) -> torch.Tensor:
        """确保矩阵满足列优先布局需求 (静态方法)"""
        return tensor.t().contiguous().t()

    def run_case(self, cfg: TrmmConfig) -> bool:
        """
        执行单次测试用例。
        G.ERR.01: 最小化 try 块。
        """
        # 1. 准备数据 (根据 side 确定 A 的维度)
        k = cfg.m
        if cfg.side.upper() == "R":
            k = cfg.n

        # 安全初始化：先 CPU 生成随机数再搬移
        mat_a_raw = torch.randn((k, k), dtype=torch.float32)
        mat_b_raw = torch.randn((cfg.m, cfg.n), dtype=torch.float32)

        # 截取三角部分逻辑拆分
        if cfg.uplo.upper() == "L":
            mat_a_raw = mat_a_raw.tril()
        else:
            mat_a_raw = mat_a_raw.triu()

        mat_a = mat_a_raw.to(self.device)
        mat_b = mat_b_raw.to(self.device)

        # 2. 核心调用 (G.ERR.01: 仅包裹算子执行及其同步)
        try:
            mat_c_npu = torch_sip.asd_blas_strmm(
                mat_a, mat_b, cfg.alpha, cfg.side, cfg.uplo, cfg.trans
            )
        except Exception as exc:
            logger.error("[%s] 算子执行崩溃: %s", cfg.name, exc)
            return False

        # 3. CPU 参考计算 (移出 try 块)
        a_ref = mat_a_raw.clone()
        b_ref = mat_b_raw.clone()

        if cfg.trans.upper() == "T":
            a_ref = a_ref.transpose(-2, -1)

        # 根据 side 执行矩阵乘法 (逻辑拆分)
        if cfg.side.upper() == "L":
            ref_c = cfg.alpha * torch.matmul(a_ref, b_ref)
        else:
            ref_c = cfg.alpha * torch.matmul(b_ref, a_ref)

        # 4. 精度比对
        npu_res_cpu = mat_c_npu.cpu()
        rtol, atol = 1e-2, 1e-2
        is_close = torch.allclose(npu_res_cpu, ref_c, rtol=rtol, atol=atol)

        status = "PASS" if is_close else "FAIL"
        logger.info("[%s] %-20s | M=%d, N=%d | Side=%s, Uplo=%s, Trans=%s",
                    status, cfg.name, cfg.m, cfg.n, cfg.side, cfg.uplo, cfg.trans)

        if not is_close:
            logger.error("      Max Diff: %.6e", torch.abs(npu_res_cpu - ref_c).max())

        return is_close


def main():
    """测试主套件入口"""
    tester = StrmmTester()
    logger.info("开始 STRMM 专项测试 (参数对象化重构版本) ...\n")

    # 使用参数对象定义测试套件
    test_suites = [
        TrmmConfig(20, 20, 2.5, "L", "L", "N", "Left_Lower_NoTrans"),
        TrmmConfig(16, 8, 1.0, "L", "U", "N", "Left_Upper_NoTrans"),
        TrmmConfig(8, 16, 1.0, "R", "L", "N", "Right_Lower_NoTrans"),
        TrmmConfig(8, 16, 2.0, "R", "U", "N", "Right_Upper_NoTrans"),
        TrmmConfig(32, 32, -1.0, "L", "L", "T", "Left_Lower_Trans"),
        TrmmConfig(32, 32, 0.5, "R", "U", "T", "Right_Upper_Trans"),
    ]

    all_passed = True
    for config in test_suites:
        if not tester.run_case(config):
            all_passed = False

    logger.info("-" * 60)
    if all_passed:
        logger.info("测试结论: ✅ 全部通过")
    else:
        logger.error("测试结论: ❌ 存在失败项")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())