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
logger = logging.getLogger("torch_sip_cal_test")


class BlasCalTester:
    """BLAS Scal 算子功能与边界测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        self.total_tests = 0
        self.passed_tests = 0

    @staticmethod
    def log_header(title: str):
        """打印测试章节标题"""
        logger.info("\n%s %s %s", "=" * 20, title, "=" * 20)

    def validate(self, scenario: str, npu_out: torch.Tensor, ref_out: torch.Tensor, atol: float = 1e-5) -> bool:
        """
        验证计算结果准确性
        """
        self.total_tests += 1

        # 1. 预检返回值是否为空 (防止链式调用崩溃)
        if npu_out is None:
            logger.error("  ✗ %s: FAILED | 算子返回值为 None", scenario)
            return False

        try:
            # 2. 搬回 CPU (处理可能的设备通信异常或 OOM)
            npu_cpu = npu_out.cpu()
            ref_cpu = ref_out.cpu()

            # 3. 校验形状一致性
            if npu_cpu.shape != ref_cpu.shape:
                logger.error("  ✗ %s: FAILED | Shape mismatch: NPU %s vs REF %s",
                             scenario, npu_cpu.shape, ref_cpu.shape)
                return False

            # 4. 精度校验
            is_close = torch.allclose(npu_cpu, ref_cpu, atol=atol)

            if is_close:
                logger.info("  ✓ %s: PASSED", scenario)
                self.passed_tests += 1
                return True

            max_diff = (npu_cpu - ref_cpu).abs().max()
            logger.error("  ✗ %s: FAILED | Max Diff: %.6e", scenario, max_diff.item())
            return False

        except RuntimeError as e:
            # 捕获框架层异常（如非法内存访问导致的同步失败）
            logger.error("  ⚠ %s: EXCEPTION | Runtime error during validation: %s", scenario, str(e))
            return False
        except Exception as e:
            logger.error("  ⚠ %s: UNKNOWN ERROR | %s", scenario, str(e))
            logger.debug(traceback.format_exc())
            return False

    def test_sscal(self):
        """测试 SSCAL (Float32 * Float 标量)"""
        BlasCalTester.log_header("Test SSCAL (Float32 * Float)")
        shape = (1024, 1024)

        try:
            x = torch.randn(shape, dtype=torch.float32, device=self.device)
            alpha = 2.5
            ref = x.cpu() * alpha

            # 显式获取返回值并增加异常防护
            out = torch_sip.asd_blas_cal(x, alpha)
            # 使用 _ = 显式接收返回值，解决 function-ret 告警 (G.FNM.07)
            _ = self.validate("SSCAL (1M elements, alpha=2.5)", out, ref)

        except Exception as e:
            self.total_tests += 1
            logger.error("  ✗ SSCAL: FAILED | Unexpected Exception: %s", str(e))

    def test_csscal(self):
        """测试 CSSCAL (Complex64 * Float 标量)"""
        BlasCalTester.log_header("Test CSSCAL (Complex64 * Float)")
        shape = (512, 512)

        try:
            real = torch.randn(shape, dtype=torch.float32)
            imag = torch.randn(shape, dtype=torch.float32)
            x = torch.complex(real, imag).to(self.device)
            alpha = 1.2

            ref = x.cpu() * alpha
            out = torch_sip.asd_blas_cal(x, alpha)
            # 使用 _ = 显式接收返回值
            _ = self.validate("CSSCAL (Complex * Real Alpha)", out, ref)

        except Exception as e:
            self.total_tests += 1
            logger.error("  ✗ CSSCAL: FAILED | Unexpected Exception: %s", str(e))

    def test_cscal(self):
        """测试 CSCAL (Complex64 * Complex64 标量)"""
        BlasCalTester.log_header("Test CSCAL (Complex64 * Complex)")
        shape = (1000,)

        try:
            real = torch.randn(shape, dtype=torch.float32)
            imag = torch.randn(shape, dtype=torch.float32)
            x = torch.complex(real, imag).to(self.device)
            alpha = complex(0.5, 1.5)

            ref = x.cpu() * alpha
            out = torch_sip.asd_blas_cal(x, alpha)
            # 使用 _ = 显式接收返回值
            _ = self.validate("CSCAL (Complex * Complex Alpha)", out, ref)

        except Exception as e:
            self.total_tests += 1
            logger.error("  ✗ CSCAL: FAILED | Unexpected Exception: %s", str(e))

    def test_exceptions(self):
        """测试异常边界：实数向量不能用复数标量缩放"""
        BlasCalTester.log_header("Test Error Guard")

        try:
            x = torch.randn((10,), dtype=torch.float32, device=self.device)
            alpha = complex(1.0, 1.0)

            logger.info("  [Info] Attempting Float32 * Complex (Should Fail)...")

            # 显式接收正常返回的情况，防止 function-ret
            _ = torch_sip.asd_blas_cal(x, alpha)

        except (RuntimeError, TypeError, ValueError) as exc:
            # 捕获到预期的 C++ TORCH_CHECK 抛出的异常
            logger.info("  ✓ Error Guard PASSED: Catch expected exception: %s", str(exc)[:60])
            self.passed_tests += 1
            self.total_tests += 1
            return
        except Exception as e:
            logger.error("  ✗ Error Guard FAILED: Caught unexpected exception type: %s", type(e))
            self.total_tests += 1
            return

        # 若未抛出异常，说明 C++ 校验逻辑缺失
        logger.error("  ✗ Error Guard FAILED: No exception raised for invalid dtypes")
        self.total_tests += 1


def main():
    """主测试流程"""
    if not torch.npu.is_available():
        logger.critical("NPU device not available. Exiting.")
        return 1

    tester = BlasCalTester()

    # 执行测试 (隐式返回 None, 无需捕获)
    tester.test_sscal()
    tester.test_csscal()
    tester.test_cscal()
    tester.test_exceptions()

    logger.info("\n" + "=" * 60)
    logger.info("Final Report: %d/%d Tests Passed", tester.passed_tests, tester.total_tests)
    logger.info("=" * 60)

    return 0 if (tester.passed_tests == tester.total_tests) else 1


if __name__ == "__main__":
    sys.exit(main())