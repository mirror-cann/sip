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
import time
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
logger = logging.getLogger("torch_sip_r2c_test")


@dataclass(frozen=True)
class R2CConfig:
    """参数对象：封装 R2C FFT 的维度与变换方向配置"""
    input_shape: Tuple[int, ...]
    is_forward: bool = True
    transform_dims: int = 1
    name: str = "Case"


class R2CFFTTester:
    """实数到复数快速傅里叶变换 (R2C FFT) 算子测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        self.total_tests = 0
        self.passed_tests = 0
        blocking_env = os.getenv("ASCEND_LAUNCH_BLOCKING", "0")
        mode_str = "Async" if blocking_env == "0" else "Blocking"
        logger.info("NPU Mode: %s (BLOCKING=%s)", mode_str, blocking_env)

    @staticmethod
    def log_section(title: str):
        """打印测试章节分割线"""
        logger.info("\n%s %s %s", "="*20, title, "="*20)

    def run_test_r2c(self, cfg: R2CConfig):
        """
        执行 R2C FFT 功能测试。
        G.ERR.01: 最小化 try 块。
        """
        # 1. 准备数据与 CPU 侧参考值
        x_real_npu, ref_val = self._prepare_test_data(cfg)

        direction_str = "Forward" if cfg.is_forward else "Inverse"
        label = f"{cfg.transform_dims}D R2C {direction_str} FFT {cfg.input_shape}"

        # 2. 核心执行 (G.ERR.01: 仅包裹算子调用与同步)
        try:
            start_ts = time.time()
            out_npu = torch_sip.asd_fft_r2c(x_real_npu, cfg.is_forward)
            duration = (time.time() - start_ts) * 1000
        except Exception as exc:
            logger.error("[%s] 运行时崩溃: %s", label, exc)
            self.total_tests += 1
            return

        # 3. 结果校验
        self._validate(label, out_npu, ref_val)
        logger.info("    - Exec Time: %.2f ms | Output Shape: %s",
                    duration, list(out_npu.shape))

    def run_type_exception_test(self):
        """验证 C++ 层的类型阻断逻辑"""
        label = "Reject Complex64 Input"
        self.total_tests += 1
        # 1. 在 CPU 上分别生成实部和虚部
        real = torch.randn((2, 16), dtype=torch.float32)
        imag = torch.randn((2, 16), dtype=torch.float32)
        # 2. 合并为复数 Tensor 并搬移到 NPU
        x_complex = torch.complex(real, imag).to(self.device)

        try:
            torch_sip.asd_fft_r2c(x_complex, True)
            logger.error("  ✗ %s: FAILED (Operator allowed complex64)", label)
        except RuntimeError as exc:
            if "must be Float32" in str(exc):
                logger.info("  ✓ %s: PASSED (Intercepted correctly)", label)
                self.passed_tests += 1
            else:
                logger.error("  ✗ %s: FAILED (Wrong error msg: %s)", label, str(exc)[:50])

    def _prepare_test_data(self, cfg: R2CConfig) -> Tuple[torch.Tensor, torch.Tensor]:
        """构造输入数据及对应的 CPU 基准结果"""
        x_real_cpu = torch.randn(cfg.input_shape, dtype=torch.float32)
        dims = tuple(range(-cfg.transform_dims, 0))

        # 计算基准
        ref = torch.fft.rfftn(x_real_cpu, dim=dims)

        # 反向逻辑：实数序列的无缩放逆变换
        if not cfg.is_forward:
            ref = ref.conj()

        return x_real_cpu.to(self.device), ref

    def _validate(self, scenario: str, npu_out: torch.Tensor, ref_out: torch.Tensor):
        """精度校验逻辑"""
        self.total_tests += 1
        npu_cpu = npu_out.cpu()

        # 对复数进行 allclose 判定
        is_close = torch.allclose(npu_cpu, ref_out, atol=1e-3, rtol=1e-3)

        if is_close:
            logger.info("  ✓ %s: PASSED", scenario)
            self.passed_tests += 1
        else:
            diff_abs = (npu_cpu - ref_out).abs()
            logger.error("  ✗ %s: FAILED | Max Abs Diff: %.6e", scenario, diff_abs.max())


def main():
    """主程序入口"""
    torch.manual_seed(1)
    tester = R2CFFTTester()

    # 1D 场景
    R2CFFTTester.log_section("1D R2C FFT Scenarios")
    tester.run_test_r2c(R2CConfig((1, 4000), is_forward=True, name="1D_Fwd"))
    tester.run_test_r2c(R2CConfig((1, 4000), is_forward=False, name="1D_Inv"))
    tester.run_test_r2c(R2CConfig((1, 2048), is_forward=True, name="1D_Fwd"))
    tester.run_test_r2c(R2CConfig((1, 2048), is_forward=False, name="1D_Inv"))
    tester.run_test_r2c(R2CConfig((16, 4000), is_forward=True, name="1D_Fwd"))
    tester.run_test_r2c(R2CConfig((16, 4000), is_forward=False, name="1D_Inv"))
    tester.run_test_r2c(R2CConfig((16, 2048), is_forward=True, name="1D_Fwd"))
    tester.run_test_r2c(R2CConfig((16, 2048), is_forward=False, name="1D_Inv"))

    # 2D 场景
    R2CFFTTester.log_section("2D R2C FFT Scenarios")
    tester.run_test_r2c(R2CConfig((8, 128, 256), transform_dims=2, name="2D_Fwd"))
    tester.run_test_r2c(R2CConfig((8, 128, 256), is_forward=False, transform_dims=2, name="2D_Inv"))

    # 3D 场景
    R2CFFTTester.log_section("3D R2C FFT Scenarios")
    tester.run_test_r2c(R2CConfig((2, 16, 32, 64), transform_dims=3, name="3D_Fwd"))
    tester.run_test_r2c(R2CConfig((2, 16, 32, 64), is_forward=False, transform_dims=3, name="3D_Inv"))

    # 异常拦截
    tester.run_type_exception_test()

    logger.info("\n" + "="*50)
    logger.info("Final Report: %d/%d Tests Passed", tester.passed_tests, tester.total_tests)
    logger.info("="*50)


if __name__ == "__main__":
    sys.exit(main())