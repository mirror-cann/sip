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
logger = logging.getLogger("torch_sip_inv_test")


@dataclass(frozen=True)
class InvertConfig:
    """参数对象：封装批量求逆的维度、精度及名称配置"""
    batch: int
    n: int
    dtype: torch.dtype = torch.complex64
    name: str = "Case"


class CmatinvBatchedTester:
    """批量复数矩阵求逆测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        blocking_env = os.getenv("ASCEND_LAUNCH_BLOCKING", "0")
        mode_str = "异步 (Async)" if blocking_env == "0" else "同步 (Blocking)"
        logger.info("当前 NPU 运行模式: %s (ASCEND_LAUNCH_BLOCKING=%s)", mode_str, blocking_env)

    @staticmethod
    def to_c64_cpu(tensor: torch.Tensor) -> torch.Tensor:
        """
        静态工具方法：处理 Complex32/64 到 CPU Complex64 的转换。
        """
        if tensor.dtype == torch.complex32:
            # Complex32 (32bit) -> View as Float32 -> CPU -> View as C32 -> To C64
            return tensor.view(torch.float32).cpu().view(torch.complex32).to(torch.complex64)
        return tensor.cpu()

    def get_safe_invertible_mat(self, cfg: InvertConfig) -> torch.Tensor:
        """
        生成可逆复数矩阵。
        通过增加对角线偏移确保矩阵非奇异。
        """
        float_dtype = torch.float16 if cfg.dtype == torch.complex32 else torch.float32
        shape = (cfg.batch, cfg.n, cfg.n)

        # 1. 在 CPU 上生成数据以规避 NPU 随机数生成限制
        real = torch.randn(shape, dtype=float_dtype)
        imag = torch.randn(shape, dtype=float_dtype)

        # 2. 增加对角线主元强度
        eye = torch.eye(cfg.n, dtype=float_dtype)
        offset = eye * 5.0
        real = real + offset.unsqueeze(0).expand(shape)

        return torch.complex(real, imag).to(self.device)

    def run_case(self, cfg: InvertConfig) -> bool:
        """
        执行单次求逆测试。
        G.ERR.01: 最小化 try 块。
        """
        # 1. 数据准备
        mat_a = self.get_safe_invertible_mat(cfg)
        a_input = mat_a.contiguous()

        # 2. 核心调用 (G.ERR.01: 仅包裹算子核心逻辑)
        try:
            a_inv_npu = torch_sip.asd_blas_cmatinv_batched(a_input)
        except Exception as exc:
            logger.error("[%s] 算子执行崩溃: %s", cfg.name, exc)
            return False

        # 3. CPU 参考计算 (linalg.inv)
        a_ref = self.to_c64_cpu(mat_a)
        ref_inv = torch.linalg.inv(a_ref)

        # 4. 精度校验
        res_cpu = self.to_c64_cpu(a_inv_npu)

        # FP16 (Complex32) 精度放宽
        rtol, atol = (2e-3, 2e-3) if cfg.dtype == torch.complex64 else (1e-2, 1e-2)
        is_close = torch.allclose(res_cpu, ref_inv, rtol=rtol, atol=atol)

        status = "PASS" if is_close else "FAIL"
        dtype_str = "C64" if cfg.dtype == torch.complex64 else "C32"
        logger.info("[%s] %-20s | %s | B=%d, N=%d", status, cfg.name, dtype_str, cfg.batch, cfg.n)

        if not is_close:
            max_diff = torch.abs(res_cpu - ref_inv).max()
            logger.error("      Max Diff: %.6e", max_diff)

        return is_close


def main():
    """主测试套件"""
    tester = CmatinvBatchedTester()
    logger.info("开始 CMATINV_BATCHED 专项求逆测试 ...\n")

    # 参数对象化定义
    test_suites = [
        InvertConfig(3, 8, name="FP32_Small"),
        InvertConfig(128, 256, name="FP32_Large"),
        InvertConfig(3, 8, dtype=torch.complex32, name="FP16_Small"),
        InvertConfig(300, 256, dtype=torch.complex32, name="FP16_Large"),
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