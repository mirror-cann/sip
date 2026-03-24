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
from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch_npu
import torch_sip

# 配置全局日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("torch_sip_cgemm")


@dataclass(frozen=True)
class GemmConfig:
    """参数对象：封装 GEMM 的所有维度和模式配置"""
    m: int
    n: int
    k: int
    trans_a: str = 'N'
    trans_b: str = 'N'
    alpha: Union[complex, float] = 1.0
    beta: Union[complex, float] = 0.0
    name: str = "Case"


class CgemmComprehensiveTester:
    def __init__(self, device: str = "npu:0"):
        self.device = device

    @staticmethod
    def get_random_scalar() -> Union[complex, float]:
        """生成随机标量逻辑"""
        real = random.uniform(-2, 2)
        if random.random() > 0.5:
            imag = random.uniform(-2, 2)
            return complex(real, imag)
        return real

    @staticmethod
    def to_column_major(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.t().contiguous().t()

    def get_complex_mat(self, shape: Tuple[int, ...]) -> torch.Tensor:
        real = torch.randn(shape, dtype=torch.float32)
        imag = torch.randn(shape, dtype=torch.float32)
        return torch.complex(real, imag).to(self.device)

    def run_case(self, cfg: GemmConfig) -> bool:
        # 1. 数据准备 (利用 cfg 属性)
        shape_a = (cfg.k, cfg.m) if cfg.trans_a.upper() != 'N' else (cfg.m, cfg.k)
        shape_b = (cfg.n, cfg.k) if cfg.trans_b.upper() != 'N' else (cfg.k, cfg.n)

        mat_a = self.get_complex_mat(shape_a)
        mat_b = self.get_complex_mat(shape_b)
        mat_c = self.get_complex_mat((cfg.m, cfg.n))
        mat_c_init = mat_c.clone()

        a_in = self.to_column_major(mat_a)
        b_in = self.to_column_major(mat_b)
        c_in = self.to_column_major(mat_c)

        # 2. 核心调用
        try:
            torch_sip.asd_blas_cgemm(
                a_in, b_in, c_in, cfg.alpha, cfg.beta, cfg.trans_a, cfg.trans_b
            )
        except Exception as exc:
            logger.error("[%s] Operator failed: %s", cfg.name, exc)
            return False

        # 3. CPU 参考计算
        def apply_op(m, op):
            op = op.upper()
            return m.t() if op == 'T' else (m.conj().t() if op == 'C' else m)

        a_ref = apply_op(mat_a.cpu(), cfg.trans_a)
        b_ref = apply_op(mat_b.cpu(), cfg.trans_b)
        ref = cfg.alpha * torch.matmul(a_ref, b_ref) + cfg.beta * mat_c_init.cpu()

        # 4. 比对
        is_close = torch.allclose(c_in.cpu(), ref, rtol=1e-3, atol=1e-3)
        if not is_close:
            logger.error("[FAIL] %s | Dim: %dx%dx%d", cfg.name, cfg.m, cfg.n, cfg.k)

        return is_close


def main():
    tester = CgemmComprehensiveTester()
    iteration = 0
    pass_count = 0

    while iteration < 100:
        iteration += 1

        # 使用辅助方法生成随机参数，保持 main 函数整洁
        m, n, k = (random.randint(1, 64) for _ in range(3))

        # 构造配置对象 (解决参数过多问题)
        config = GemmConfig(
            m=m, n=n, k=k,
            alpha=tester.get_random_scalar(),
            beta=tester.get_random_scalar(),
            trans_a=random.choice(['N', 'T', 'C']),
            trans_b=random.choice(['N', 'T', 'C']),
            name=f"Iter_{iteration}"
        )

        if tester.run_case(config):
            pass_count += 1

        if (iteration - 1) % 100 == 0:
            logger.info("Progress: %d/100 cases finished.", iteration)

    return 0 if pass_count == iteration else 1


if __name__ == "__main__":
    sys.exit(main())