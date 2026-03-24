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
from typing import Tuple, Dict, Any

import torch
import torch_npu
import torch_sip

# 配置全局日志 (G.LOG.02)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("torch_sip_c2r_test")


@dataclass(frozen=True)
class C2RConfig:
    """参数对象：封装 C2R FFT 的维度与变换方向配置"""
    real_shape: Tuple[int, ...]
    is_forward: bool = True
    transform_dims: int = 1
    name: str = "Case"


class C2RFFTTester:
    """复数到实数快速傅里叶变换 (C2R FFT) 算子测试类"""

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
        logger.info("\n%s %s %s", "=" * 20, title, "=" * 20)

    def run_test_c2r(self, cfg: C2RConfig):
        """
        执行 C2R FFT 功能测试。
        G.ERR.01: 最小化 try 块。
        """
        # 1. 准备数据与参考值 (CPU 侧完成)
        x_complex_npu, ref_val = self._prepare_test_data(cfg)

        direction_str = "Forward" if cfg.is_forward else "Inverse"
        label = f"{cfg.transform_dims}D C2R {direction_str} FFT {cfg.real_shape}"

        # 2. 核心执行 (G.ERR.01: 仅包裹算子调用与同步)
        try:
            start_ts = time.time()
            out_npu = torch_sip.asd_fft_c2r(x_complex_npu, cfg.is_forward)
            duration = (time.time() - start_ts) * 1000
        except Exception as exc:
            logger.error("[%s] 运行时崩溃: %s", label, exc)
            self.total_tests += 1
            return

        # 3. 校验
        self._validate(label, out_npu, ref_val)
        logger.info("    - Exec Time: %.2f ms | Output Shape: %s",
                    duration, list(out_npu.shape))

    def run_type_exception_test(self):
        """验证 C++ 层的类型阻断逻辑"""
        label = "Reject Float32 Input"
        self.total_tests += 1
        x_float = torch.randn((2, 16), dtype=torch.float32, device=self.device)

        try:
            torch_sip.asd_fft_c2r(x_float, True)
            logger.error("  ✗ %s: FAILED (Operator allowed float32)", label)
        except RuntimeError as exc:
            if "must be ComplexFloat" in str(exc):
                logger.info("  ✓ %s: PASSED (Intercepted correctly)", label)
                self.passed_tests += 1
            else:
                logger.error("  ✗ %s: FAILED (Wrong error msg: %s)", label, str(exc)[:50])

    def _prepare_test_data(self, cfg: C2RConfig) -> Tuple[torch.Tensor, torch.Tensor]:
        """构造具有共轭对称性的输入数据及参考结果"""
        # C2R 需要输入满足共轭对称性，最简单的方法是对实数做 RFFT 得到输入
        x_real_cpu = torch.randn(cfg.real_shape, dtype=torch.float32)
        dims = tuple(range(-cfg.transform_dims, 0))

        # 得到 Complex64 频域数据
        x_complex_cpu = torch.fft.rfftn(x_real_cpu, dim=dims)

        # 计算基准 (ASD 算子通常不包含 1/N 归一化)
        if cfg.is_forward:
            ref = torch.fft.irfftn(x_complex_cpu.conj(), dim=dims)
        else:
            ref = torch.fft.irfftn(x_complex_cpu, dim=dims)

        # 补回缩放系数
        scale = 1.0
        for d in dims:
            scale *= cfg.real_shape[d]
        ref = ref * scale

        return x_complex_cpu.to(self.device), ref

    def _validate(self, scenario: str, npu_out: torch.Tensor, ref_out: torch.Tensor):
        """精度校验逻辑"""
        self.total_tests += 1
        npu_cpu = npu_out.cpu()
        is_close = torch.allclose(npu_cpu, ref_out, atol=1e-2, rtol=1e-2)

        if is_close:
            logger.info("  ✓ %s: PASSED", scenario)
            self.passed_tests += 1
        else:
            max_diff = (npu_cpu - ref_out).abs().max()
            logger.error("  ✗ %s: FAILED | Max Diff: %.6e", scenario, max_diff)


def main():
    """主程序入口"""
    tester = C2RFFTTester()

    # 1D 场景
    C2RFFTTester.log_section("1D C2R FFT Scenarios")
    tester.run_test_c2r(C2RConfig((16, 4096), name="1D_Fwd"))
    tester.run_test_c2r(C2RConfig((16, 1024), is_forward=False, name="1D_Inv"))
    tester.run_test_c2r(C2RConfig((16, 2048), name="1D_Fwd"))
    tester.run_test_c2r(C2RConfig((16, 2048), is_forward=False, name="1D_Inv"))
    tester.run_test_c2r(C2RConfig((1, 2048), name="1D_Fwd"))
    tester.run_test_c2r(C2RConfig((1, 2048), is_forward=False, name="1D_Inv"))

    # 2D 场景
    C2RFFTTester.log_section("2D C2R FFT Scenarios")
    tester.run_test_c2r(C2RConfig((2, 128, 128), transform_dims=2, name="2D_Fwd"))
    tester.run_test_c2r(C2RConfig((8, 4, 1024), is_forward=False, transform_dims=2, name="2D_Inv"))

    # 3D 场景
    C2RFFTTester.log_section("3D C2R FFT Scenarios")
    tester.run_test_c2r(C2RConfig((2, 16, 32, 64), transform_dims=3, name="3D_Fwd"))
    tester.run_test_c2r(C2RConfig((2, 16, 32, 64), is_forward=False, transform_dims=3, name="3D_Inv"))

    # 异常拦截
    tester.run_type_exception_test()

    logger.info("\n" + "=" * 50)
    logger.info("Final Report: %d/%d Tests Passed", tester.passed_tests, tester.total_tests)
    logger.info("=" * 50)


if __name__ == "__main__":
    sys.exit(main())
