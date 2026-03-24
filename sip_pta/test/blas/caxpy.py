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
import sys
import time
import traceback

import torch
import torch_npu
import torch_sip

# 配置全局日志 (G.LOG.02)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("torch_sip_caxpy_test")


class CaxpyTester:
    """CAXPY 算子功能测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        self.total_tests = 0
        self.passed_tests = 0

    @staticmethod
    def log_header(title: str):
        """打印测试章节标题"""
        logger.info("\n%s %s %s", "=" * 25, title, "=" * 25)

    def validate(self, scenario: str, x: torch.Tensor, y_init: torch.Tensor,
                 alpha: complex, y_npu: torch.Tensor) -> bool:
        """
        验证计算结果准确性 (遵循 G.FNM.07: 处理潜在同步异常与资源风险)
        """
        self.total_tests += 1
        atol = 1e-5

        try:
            # 1. 显式处理数据同步 (NPU -> CPU)
            # 对于 In-place 操作，y_npu 可能在算子失败时包含无效内存
            npu_res = y_npu.cpu()
            ref_x = x.cpu()
            ref_y_init = y_init.cpu()

            # 2. CPU 参考计算: y = alpha * x + y
            ref = (alpha * ref_x) + ref_y_init

            # 3. 结果一致性校验
            is_close = torch.allclose(npu_res, ref, atol=atol)

            if is_close:
                logger.info("  ✓ %s: PASSED", scenario)
                self.passed_tests += 1
                return True

            max_diff = (npu_res - ref).abs().max()
            logger.error("  ✗ %s: FAILED | 结果不匹配, 最大偏差: %.6e", scenario, max_diff.item())
            return False

        except RuntimeError as e:
            # 捕获 NPU 同步失败、内存访问越界等异常
            logger.error("  ⚠ %s: EXCEPTION | 同步或验证过程失败: %s", scenario, str(e))
            return False
        except Exception as e:
            logger.error("  ⚠ %s: UNKNOWN ERROR | %s", scenario, str(e))
            logger.debug(traceback.format_exc())
            return False

    def run_test(self, name: str, shape: tuple, alpha: complex):
        """执行单次 CAXPY 测试"""
        # 1. 安全初始化 (遵循 G.LOG.02 记录关键步骤)
        def generate_complex(s):
            real = torch.randn(s, dtype=torch.float32)
            imag = torch.randn(s, dtype=torch.float32)
            return torch.complex(real, imag).to(self.device)

        try:
            x = generate_complex(shape)
            y = generate_complex(shape)
            y_init = y.clone()  # 保存初始值用于验证
        except Exception as e:
            logger.error("  ⚠ %s: 初始化数据失败 (可能是显存不足): %s", name, e)
            self.total_tests += 1
            return

        # 2. 核心调用 (G.ERR.01: 最小化 try 块范围)
        try:
            t0 = time.perf_counter()
            # CAXPY 通常是 In-place 修改 y
            torch_sip.asd_blas_caxpy(x, y, alpha)
            duration_ms = (time.perf_counter() - t0) * 1000

            # 3. 验证结果并处理其返回值 (G.FNM.07)
            success = self.validate(name, x, y_init, alpha, y)
            if success:
                logger.info("    - Alpha: %s | 时间: %.2f ms", alpha, duration_ms)

        except Exception as exc:
            # 记录算子执行层面的异常
            logger.error("  ✗ %s: 算子执行崩溃 -> %s", name, exc)
            self.total_tests += 1


def main():
    """测试主流程入口"""
    if not torch.npu.is_available():
        logger.critical("NPU 环境不可用，退出测试。")
        return 1

    logger.info("torch_sip CAXPY (Complex AXPY) 测试开始")
    tester = CaxpyTester()

    # 1. 标准复数测试
    CaxpyTester.log_header("Case 1: Standard Complex Alpha")
    tester.run_test("Complex Alpha (1024)", (1024,), complex(1.5, -2.0))

    # 2. 类型转换测试
    CaxpyTester.log_header("Case 2: Alpha Type Conversion")
    tester.run_test("Float Alpha (512x512)", (512, 512), 2.5)
    tester.run_test("Int Alpha (1M)", (1000000,), 5)

    # 3. 边界测试
    CaxpyTester.log_header("Case 3: Boundary Values")
    tester.run_test("Pure Imaginary (128)", (128,), complex(0, 1))
    tester.run_test("Zero Alpha", (1024,), complex(0, 0))

    # 汇总报告
    logger.info("\n" + "=" * 60)
    logger.info("CAXPY 测试汇总: %d/%d 通过", tester.passed_tests, tester.total_tests)
    logger.info("=" * 60)

    return 0 if (tester.passed_tests == tester.total_tests) else 1


if __name__ == "__main__":
    # 明确处理 main 的返回值 (G.FNM.07)
    sys.exit(main())