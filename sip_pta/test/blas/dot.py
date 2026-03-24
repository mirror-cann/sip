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
from typing import Optional

import torch
import torch_npu
import torch_sip

# 配置全局日志 (G.LOG.02)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("torch_sip_dot_test")


@dataclass(frozen=True)
class DotConfig:
    """参数对象：封装向量内积的维度、类型及共轭配置"""
    n: int
    dtype: torch.dtype
    conjugate: bool = False
    name: str = "Case"


class DotTester:
    """向量内积 (Dot) 算子测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        blocking_env = os.getenv("ASCEND_LAUNCH_BLOCKING", "0")
        mode_str = "异步 (Async)" if blocking_env == "0" else "同步 (Blocking)"
        logger.info("当前 NPU 模式: %s (BLOCKING=%s)", mode_str, blocking_env)

    @staticmethod
    def get_tensor(n: int, dtype: torch.dtype, device: str) -> torch.Tensor:
        """静态方法：安全生成测试 Tensor (先 CPU 后搬移)"""
        if dtype == torch.complex64:
            real = torch.randn(n, dtype=torch.float32)
            imag = torch.randn(n, dtype=torch.float32)
            return torch.complex(real, imag).to(device)

        return torch.randn(n, dtype=torch.float32).to(device)

    def run_case(self, cfg: DotConfig) -> bool:
        """
        执行单次测试用例。
        G.ERR.01: 最小化 try 块。
        """
        # 1. 准备数据
        vec_x = self.get_tensor(cfg.n, cfg.dtype, self.device)
        vec_y = self.get_tensor(cfg.n, cfg.dtype, self.device)

        # 2. 核心调用 (G.ERR.01: 仅包裹算子执行及其同步)
        try:
            res_npu = torch_sip.asd_blas_dot(vec_x, vec_y, cfg.conjugate)
        except Exception as exc:
            logger.error("[%s] 算子执行崩溃: %s", cfg.name, exc)
            return False

        # 3. CPU 参考计算 (移出 try 块)
        x_cpu = vec_x.cpu()
        y_cpu = vec_y.cpu()

        if cfg.dtype == torch.float32:
            ref = torch.dot(x_cpu, y_cpu)
            mode_name = "SDOT"
        else:
            # 复数点积逻辑拆分
            if cfg.conjugate:
                # CDOTC 定义: dotc(x, y) = sum(conj(x_i) * y_i)
                ref = (x_cpu.conj() * y_cpu).sum()
                mode_name = "CDOTC"
            else:
                # CDOTU 定义: dotu(x, y) = sum(x_i * y_i)
                ref = (x_cpu * y_cpu).sum()
                mode_name = "CDOTU"

        # 4. 验证结果
        is_close = torch.allclose(res_npu.cpu(), ref, rtol=1e-3, atol=1e-3)

        status = "PASS" if is_close else "FAIL"
        logger.info("[%s] %-20s | %s | Conj=%s | N=%d",
                    status, cfg.name, mode_name, cfg.conjugate, cfg.n)

        if not is_close:
            logger.error("      Max Diff: %.6e", torch.abs(res_npu.cpu() - ref).max())

        return is_close


def main():
    """测试主入口"""
    tester = DotTester()
    logger.info("开始 Dot (SDOT/CDOTU/CDOTC) 专项测试 ...\n")

    # 使用参数对象定义测试套件
    test_suites = [
        DotConfig(1024, torch.float32, name="Float_Dot"),
        DotConfig(2048, torch.complex64, conjugate=False, name="Complex_Dotu"),
        DotConfig(2048, torch.complex64, conjugate=True, name="Complex_Dotc"),
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