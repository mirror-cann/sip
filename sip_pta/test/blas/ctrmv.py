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
from typing import Tuple, Optional, Any, Dict

import torch
import torch_npu
import torch_sip

# 配置全局日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("torch_sip_cgemv_batched")


@dataclass(frozen=True)
class BatchedGemvConfig:
    """参数对象：封装 Batched CGEMV 的所有维度与模式配置"""
    batch: int
    m: int
    n: int
    alpha: float
    beta: float
    trans: str = "N"
    dtype: torch.dtype = torch.complex64
    name: str = "Case"


class CgemvBatchedTester:
    """Batched CGEMV 算子测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        blocking_env = os.getenv("ASCEND_LAUNCH_BLOCKING", "0")
        mode_str = "异步 (Async)" if blocking_env == "0" else "同步 (Blocking)"
        logger.info("当前 NPU 运行模式: %s (ASCEND_LAUNCH_BLOCKING=%s)", mode_str, blocking_env)

    @staticmethod
    def get_complex_tensor(shape: Tuple[int, ...], device: str, dtype: torch.dtype):
        """生成复数 Tensor 辅助函数"""
        f_dtype = torch.float16 if dtype == torch.complex32 else torch.float32
        real = torch.randn(shape, dtype=f_dtype, device=device)
        imag = torch.randn(shape, dtype=f_dtype, device=device)
        return torch.complex(real, imag)

    def run_case(self, cfg: BatchedGemvConfig) -> bool:
        """
        执行单次测试。
        参数已封装至 cfg 对象。
        """
        x_len, y_len = cfg.n, cfg.m

        # 1. 数据准备
        mat_a = self.get_complex_tensor((cfg.batch, cfg.m, cfg.n), self.device, cfg.dtype)
        vec_x = self.get_complex_tensor((cfg.batch, x_len), self.device, cfg.dtype)
        vec_y = self.get_complex_tensor((cfg.batch, y_len), self.device, cfg.dtype)

        if cfg.dtype == torch.complex32:
            vec_y_init = vec_y.view(torch.int32).clone().view(torch.complex32)
        else:
            vec_y_init = vec_y.clone()

        # 2. 构建动态参数 (已拆分为多行)
        kwargs: Dict[str, Any] = {}
        if cfg.alpha is not None:
            kwargs['alpha'] = cfg.alpha
        if cfg.beta is not None:
            kwargs['beta'] = cfg.beta
        if cfg.trans is not None:
            kwargs['trans'] = cfg.trans

        # 3. NPU 计算 (G.ERR.01)
        try:
            torch_sip.asd_blas_cgemv_batched(mat_a, vec_x, vec_y, **kwargs)
        except Exception as exc:
            logger.error("[%s] 算子崩溃: %s", cfg.name, exc)
            return False

        # 4. CPU 参考计算 (保持原始路径)
        if cfg.dtype == torch.complex32:
            a_ref = mat_a.view(torch.float32).cpu().view(torch.complex32).to(torch.complex64)
            x_ref = vec_x.view(torch.float32).cpu().view(torch.complex32).to(torch.complex64).unsqueeze(-1)
            y_ref = vec_y_init.view(torch.float32).cpu().view(torch.complex32).to(torch.complex64).unsqueeze(-1)
        else:
            a_ref = mat_a.cpu()
            x_ref = vec_x.cpu().unsqueeze(-1)
            y_ref = vec_y_init.cpu().unsqueeze(-1)

        if cfg.trans.upper() == "T":
            a_ref = a_ref.transpose(1, 2)
        elif cfg.trans.upper() == "C":
            a_ref = a_ref.conj().transpose(1, 2)

        ref = torch.baddbmm(y_ref, a_ref, x_ref, alpha=cfg.alpha, beta=cfg.beta).squeeze(-1)

        # 5. 校验
        rtol, atol = (1e-4, 1e-4) if cfg.dtype == torch.complex64 else (1e-2, 1e-2)
        is_close = torch.allclose(vec_y.cpu().to(torch.complex64), ref, rtol=rtol, atol=atol)

        status = "PASS" if is_close else "FAIL"
        dtype_str = "C64" if cfg.dtype == torch.complex64 else "C32"
        logger.info("[%s] %-20s | %s | B=%d, M=%d, N=%d",
                    status, cfg.name, dtype_str, cfg.batch, cfg.m, cfg.n)

        if not is_close:
            logger.error("Max Diff: %.6f", (vec_y.cpu().to(torch.complex64) - ref).abs().max())

        return is_close


def main():
    """主程序"""
    tester = CgemvBatchedTester()

    # 使用参数对象定义用例，解决 run_case 参数过多问题
    test_suites = [
        BatchedGemvConfig(3, 16, 8, 1.0, 0.0, name="FP32_Normal"),
        BatchedGemvConfig(2, 32, 32, 1.0, 0.0, trans="C", name="FP32_Conj"),
        BatchedGemvConfig(3, 16, 8, 1.0, 0.0, dtype=torch.complex32, name="FP16_Normal"),
        BatchedGemvConfig(2, 32, 32, 1.0, 0.0, trans="C", dtype=torch.complex32, name="FP16_Conj"),
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