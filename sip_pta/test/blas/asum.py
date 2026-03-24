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
import random
import sys
import time

import torch
import torch_npu

import torch_sip

# 配置全局日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("torch_sip_stability")


def run_test_case(name, x_npu, verbose=True):
    """
    单次测试执行逻辑。
    G.ERR.01: try 块内只包含可能抛出异常的算子调用。
    """
    # 1. 准备参考值 (CPU上计算)
    x_cpu = x_npu.cpu()
    if x_npu.is_complex():
        # Scasum 逻辑
        ref_val = (torch.abs(x_cpu.real).sum() + torch.abs(x_cpu.imag).sum()).item()
    else:
        # Sasum 逻辑
        ref_val = torch.abs(x_cpu).sum().item()

    # 2. 核心调用 (最小化 try 块，仅包裹算子执行)
    try:
        result = torch_sip.asd_blas_asum(x_npu)
    except Exception as exc:
        logger.error("  ✗ [%s] Execution Error: %s", name, exc)
        return False

    # 3. 结果验证 (逻辑处理移出 try 块)
    result_val = result.item()
    diff = abs(result_val - ref_val)
    # 考虑到浮点数累加误差的阈值
    tol = ref_val * 1e-4 + 1e-5
    is_passed = diff <= tol

    if verbose or not is_passed:
        status = "✓ PASSED" if is_passed else "✗ FAILED"
        logger.info(
            "  [%s] Dtype: %s | N: %d | Result: %.4f | Ref: %.4f | Status: %s",
            name, x_npu.dtype, x_npu.numel(), result_val, ref_val, status
        )

    return is_passed


def get_random_case(device):
    """
    生成随机测试用例。
    对于 Complex64，采用先 CPU 创建再搬移至 NPU 的安全策略。
    """
    case_types = ["float_std", "complex_std", "float_sliced", "complex_sliced"]
    case_type = random.choice(case_types)
    size = random.randint(10, 256)

    if case_type == "float_std":
        return "Float32_Std", torch.randn(size, dtype=torch.float32, device=device)

    if case_type == "complex_std":
        # 规避 NPU 直接生成复数随机数的潜在问题
        r = torch.randn(size, dtype=torch.float32)
        i = torch.randn(size, dtype=torch.float32)
        return "Complex64_Std", torch.complex(r, i).npu()

    if case_type == "float_sliced":
        x = torch.randn(size * 2, dtype=torch.float32, device=device)
        return "Float32_Sliced", x[::2]

    if case_type == "complex_sliced":
        r = torch.randn(size * 2, dtype=torch.float32)
        i = torch.randn(size * 2, dtype=torch.float32)
        x_c = torch.complex(r, i).npu()
        return "Complex64_Sliced", x_c[::2]

    return "Unknown", None


def main():
    """压力测试主循环"""
    device = "npu:0"
    num_iterations = 1

    logger.info("=" * 60)
    logger.info("Stability Stress Test: Running %d iterations", num_iterations)
    logger.info("=" * 60)

    pass_count = 0
    start_time = time.time()

    try:
        for i in range(1, num_iterations + 1):
            name, x = get_random_case(device)

            # 定期清理显存碎片
            if i % 1000 == 0:
                torch_npu.npu.empty_cache()

            # 打印策略：前5次必显，之后每20次采样
            is_verbose = (i <= 5 or i % 20 == 0)
            success = run_test_case(f"Iter {i:03d}_{name}", x, verbose=is_verbose)

            if success:
                pass_count += 1
            else:
                logger.error("!!! STABILITY FAILURE at iteration %d !!!", i)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")

    end_time = time.time()

    logger.info("\n" + "=" * 60)
    logger.info("Final Stability Report: %d/%d passed", pass_count, num_iterations)
    logger.info("Total Time: %.2f s", end_time - start_time)
    logger.info("=" * 60)

    return 0 if (pass_count == num_iterations) else 1


if __name__ == "__main__":
    sys.exit(main())