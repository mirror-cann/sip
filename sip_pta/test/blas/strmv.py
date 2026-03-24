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
logger = logging.getLogger("torch_sip_strmv_test")


@dataclass(frozen=True)
class TrmvConfig:
    """参数对象：封装 STRMV 的所有维度、模式及名称配置"""
    n: int
    uplo: str = "L"
    trans: str = "N"
    diag: str = "N"
    name: str = "Case"


class StrmvTester:
    """单精度三角矩阵-向量乘法 (STRMV) 算子测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        blocking_env = os.getenv("ASCEND_LAUNCH_BLOCKING", "0")
        mode_str = "异步 (Async)" if blocking_env == "0" else "同步 (Blocking)"
        logger.info("当前 NPU 运行模式: %s (ASCEND_LAUNCH_BLOCKING=%s)", mode_str, blocking_env)

    @staticmethod
    def to_column_major(tensor: torch.Tensor) -> torch.Tensor:
        """确保矩阵满足列优先布局需求 (静态方法)"""
        return tensor.t().contiguous().t()

    def run_case(self, cfg: TrmvConfig) -> bool:
        """
        版本 1: A 由行存储转为列存储。
        G.ERR.01: 最小化 try 块。
        """
        # 1. 准备数据
        mat_a = torch.randn((cfg.n, cfg.n), dtype=torch.float32, device=self.device)
        vec_x = torch.randn((cfg.n,), dtype=torch.float32, device=self.device)
        mat_a = self.to_column_major(mat_a)

        # 2. CPU 参考计算逻辑 (逻辑拆分)
        a_ref = mat_a.cpu().clone()
        if cfg.uplo.upper() == "L":
            a_ref = torch.tril(a_ref)
        else:
            a_ref = torch.triu(a_ref)

        if cfg.diag.upper() == "U":
            a_ref.fill_diagonal_(1.0)

        if cfg.trans.upper() == "T":
            a_ref = a_ref.t()

        ref_res = torch.mv(a_ref, vec_x.cpu())

        # 3. 核心调用 (G.ERR.01: 仅包裹原地修改的算子执行)
        try:
            torch_sip.asd_blas_strmv(mat_a, vec_x, cfg.uplo, cfg.trans, cfg.diag)
        except Exception as exc:
            logger.error("[%s] V1 算子崩溃: %s", cfg.name, exc)
            return False

        # 4. 精度比对
        is_close = torch.allclose(vec_x.cpu(), ref_res, rtol=1e-3, atol=1e-3)
        status = "PASS" if is_close else "FAIL"
        logger.info("[%s] V1 | N=%d | %s, %s, %s | %s",
                    cfg.name, cfg.n, cfg.uplo, cfg.trans, cfg.diag, status)
        return is_close

    def run_case_v2(self, cfg: TrmvConfig) -> bool:
        """
        版本 2: A 不转为列存储, 使用 A^T 传入算子。
        G.ERR.01: 最小化 try_块。
        """
        # 1. 准备数据
        mat_a_raw = torch.randn((cfg.n, cfg.n), dtype=torch.float32, device=self.device)
        vec_x = torch.randn((cfg.n,), dtype=torch.float32, device=self.device)

        # 2. CPU 参考计算逻辑
        a_ref = mat_a_raw.cpu().clone()
        mat_a_npu = mat_a_raw.t()  # 模拟转置输入

        if cfg.uplo.upper() == "L":
            a_ref = torch.triu(a_ref)  # 转置后 L 变 U
        else:
            a_ref = torch.tril(a_ref)  # 转置后 U 变 L

        if cfg.diag.upper() == "U":
            a_ref.fill_diagonal_(1.0)

        if cfg.trans.upper() != "T":
            a_ref = a_ref.t()

        ref_res = torch.mv(a_ref, vec_x.cpu())

        # 3. 核心调用
        try:
            torch_sip.asd_blas_strmv(mat_a_npu, vec_x, cfg.uplo, cfg.trans, cfg.diag)
        except Exception as exc:
            logger.error("[%s] V2 算子崩溃: %s", cfg.name, exc)
            return False

        # 4. 精度比对
        is_close = torch.allclose(vec_x.cpu(), ref_res, rtol=1e-3, atol=1e-3)
        status = "PASS" if is_close else "FAIL"
        logger.info("[%s] V2 | N=%d | %s, %s, %s | %s",
                    cfg.name, cfg.n, cfg.uplo, cfg.trans, cfg.diag, status)
        return is_close


def main():
    """测试主套件入口"""
    tester = StrmvTester()
    logger.info("开始 STRMV 专项测试 (参数对象化重构版本) ...\n")

    # 定义测试套件
    test_suites = [
        TrmvConfig(4, "U", "N", "N", "Upper_Normal_NonUnit"),
        TrmvConfig(8, "L", "N", "N", "Lower_Normal_NonUnit"),
        TrmvConfig(16, "U", "T", "N", "Upper_Trans_NonUnit"),
        TrmvConfig(32, "L", "T", "U", "Lower_Trans_Unit"),
        TrmvConfig(64, "U", "N", "U", "Upper_Normal_Unit"),
    ]

    all_passed = True

    # 批量执行 V1 测试
    logger.info("--- 执行 V1 (Column-Major) 测试 ---")
    for config in test_suites:
        if not tester.run_case(config):
            all_passed = False

    # 批量执行 V2 测试
    logger.info("\n--- 执行 V2 (Row-Major/Transposed) 测试 ---")
    for config in test_suites:
        if not tester.run_case_v2(config):
            all_passed = False

    logger.info("-" * 60)
    if all_passed:
        logger.info("测试结论: ✅ 全部通过")
    else:
        logger.error("测试结论: ❌ 存在失败项")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())