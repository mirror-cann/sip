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
logger = logging.getLogger("torch_sip_interp_coeff")


@dataclass(frozen=True)
class InterpConfig:
    """参数对象：封装插值算子的维度、类型及名称"""
    batch: int
    n_rs: int
    total_subcarrier: int
    n_signal: int = 14
    dtype: torch.dtype = torch.complex64
    name: str = "Case"


class InterpWithCoeffTester:
    """带系数的线性插值 (InterpWithCoeff) 算子测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        blocking_env = os.getenv("ASCEND_LAUNCH_BLOCKING", "0")
        mode_str = "异步 (Async)" if blocking_env == "0" else "同步 (Blocking)"
        logger.info("当前 NPU 模式: %s (BLOCKING=%s)", mode_str, blocking_env)

    # --- Static Methods ---

    @staticmethod
    def log_section(title: str):
        """静态工具方法：打印测试章节分割线"""
        logger.info("\n" + "="*20 + f" {title} " + "="*20)

    # --- Public Test APIs ---

    def run_case(self, cfg: InterpConfig) -> bool:
        """
        执行单次插值测试用例。
        G.ERR.01: 最小化 try 块。
        """
        out_m = cfg.n_signal - cfg.n_rs

        # 1. 准备数据 (安全初始化：CPU 生成 -> 搬移)
        tensor_x = self._get_complex_tensor((cfg.batch, cfg.n_rs, cfg.total_subcarrier), cfg.dtype)
        tensor_coeff = self._get_complex_tensor((cfg.batch, out_m, cfg.n_rs), cfg.dtype)

        # 2. 构建 CPU Reference (高精度计算)
        x_ref = tensor_x.cpu().to(torch.complex64)
        coeff_ref = tensor_coeff.cpu().to(torch.complex64)
        ref_out = torch.bmm(coeff_ref, x_ref)

        # 3. 核心执行 (G.ERR.01: 仅包裹算子核心逻辑)
        try:
            npu_out = torch_sip.asd_interp_with_coeff(tensor_x, tensor_coeff)
        except Exception as exc:
            logger.error("[%s] 算子执行崩溃: %s", cfg.name, exc)
            return False

        # 4. 精度比对
        npu_res_c64 = npu_out.cpu().to(torch.complex64)
        rtol, atol = (1e-3, 1e-3) if cfg.dtype == torch.complex64 else (1e-2, 1e-2)
        is_close = torch.allclose(npu_res_c64, ref_out, rtol=rtol, atol=atol)

        # 5. 结果打印
        status = "PASS" if is_close else "FAIL"
        dtype_str = "C64" if cfg.dtype == torch.complex64 else "C32"
        logger.info("[%s] %-20s | %s | B=%d, nRs=%d, SubC=%d",
                    status, cfg.name, dtype_str, cfg.batch, cfg.n_rs, cfg.total_subcarrier)

        if not is_close:
            max_err = (npu_res_c64 - ref_out).abs().max().item()
            logger.error("      -> Max Absolute Error: %.5f", max_err)

        return is_close

    # --- Internal Helpers ---

    def _get_complex_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """辅助方法：安全生成复数测试张量"""
        f_dtype = torch.float16 if dtype == torch.complex32 else torch.float32
        real = torch.randn(shape, dtype=f_dtype)
        imag = torch.randn(shape, dtype=f_dtype)
        return torch.complex(real, imag).to(self.device)


def main() -> int:
    """
    测试主套件入口。
    返回: 0 代表全部通过, 1 代表有失败项。
    """
    tester = InterpWithCoeffTester()
    all_passed = True

    # 使用参数对象定义测试套件
    test_suites = [
        InterpConfig(1, 2, 32, name="SingleBatch_C64"),
        InterpConfig(4, 4, 64, name="MultiBatch_C64"),
        InterpConfig(8, 2, 128, name="LargeSubC_C64"),
        InterpConfig(1, 2, 32, dtype=torch.complex32, name="SingleBatch_C32"),
        InterpConfig(4, 4, 64, dtype=torch.complex32, name="MultiBatch_C32"),
    ]

    InterpWithCoeffTester.log_section("信道估计插值算子连通性测试")

    for config in test_suites:
        # 只要有一个失败，结果即为失败
        if not tester.run_case(config):
            all_passed = False

    logger.info("-" * 60)
    if all_passed:
        logger.info("测试结论: ✅ 全部通过")
        return 0
    else:
        logger.error("测试结论: ❌ 存在失败项")
        return 1


if __name__ == "__main__":
    sys.exit(main())
