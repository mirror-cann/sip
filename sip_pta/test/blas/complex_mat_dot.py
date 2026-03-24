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
from typing import Tuple

import torch
import torch_npu
import torch_sip

# 配置全局日志 (G.LOG.02)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("torch_sip_matdot_test")


@dataclass(frozen=True)
class MatDotConfig:
    """参数对象：封装矩阵点乘的维度及用例名称配置"""
    m: int
    n: int
    name: str = "Case"


class ComplexMatDotTester:
    """复数矩阵逐元素点乘测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        blocking_env = os.getenv("ASCEND_LAUNCH_BLOCKING", "0")
        mode_str = "异步 (Async)" if blocking_env == "0" else "同步 (Blocking)"
        logger.info("当前 NPU 运行模式: %s (ASCEND_LAUNCH_BLOCKING=%s)", mode_str, blocking_env)

    @staticmethod
    def get_complex_tensor(shape: Tuple[int, ...], device: str) -> torch.Tensor:
        """
        静态方法：安全生成复数 Tensor。
        整改：先在 CPU 生成随机数再搬移至目标设备，确保复数随机生成的稳定性。
        """
        real = torch.randn(shape, dtype=torch.float32)
        imag = torch.randn(shape, dtype=torch.float32)
        return torch.complex(real, imag).to(device)

    def run_case(self, cfg: MatDotConfig) -> bool:
        """
        执行单次测试用例。
        G.ERR.01: 最小化 try 块，仅包裹算子核心调用逻辑。
        """
        # 1. 准备数据
        mat_x = self.get_complex_tensor((cfg.m, cfg.n), self.device)
        mat_y = self.get_complex_tensor((cfg.m, cfg.n), self.device)

        # 备份用于 CPU 参考计算
        mat_x_cpu = mat_x.cpu()
        mat_y_cpu = mat_y.cpu()

        # 确保内存连续
        x_in = mat_x.contiguous()
        y_in = mat_y.contiguous()

        # 2. 核心调用 (G.ERR.01: 仅包裹算子执行及其同步)
        try:
            out_npu = torch_sip.asd_blas_complex_mat_dot(x_in, y_in)
        except Exception as exc:
            logger.error("[%s] 算子执行崩溃: %s", cfg.name, exc)
            return False

        # 3. CPU 参考计算 (移出 try 块)
        ref = mat_x_cpu * mat_y_cpu

        # 4. 精度比对与验证
        out_cpu = out_npu.cpu()
        is_close = torch.allclose(out_cpu, ref, rtol=1e-4, atol=1e-4)

        status = "PASS" if is_close else "FAIL"
        logger.info("[%s] %-20s | M=%d, N=%d", status, cfg.name, cfg.m, cfg.n)

        if not is_close:
            max_diff = torch.abs(out_cpu - ref).max()
            logger.error("      Max Diff: %.6e", max_diff)

        return is_close


def main():
    """测试主套件入口"""
    tester = ComplexMatDotTester()
    logger.info("开始 ComplexMatDot 专项测试 (参数对象化重构版本) ...\n")

    # 使用参数对象定义测试套件
    test_suites = [
        MatDotConfig(8, 4, "Small_Matrix"),
        MatDotConfig(16, 16, "Square_Matrix"),
        MatDotConfig(128, 64, "Large_Rectangle"),
        MatDotConfig(256, 256, "Larger_Square"),
    ]

    all_passed = True
    for config in test_suites:
        if not tester.run_case(config):
            all_passed = False

    logger.info("-" * 50)
    if all_passed:
        logger.info("测试结论: ✅ 全部通过")
    else:
        logger.error("测试结论: ❌ 存在失败项")

    return 0 if all_passed else 1


if __name__ == "__main__":
    # 整个脚本唯一的退出入口，负责返回主流程状态码
    sys.exit(main())