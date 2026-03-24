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
import random
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
logger = logging.getLogger("torch_sip_iamax_test")


@dataclass(frozen=True)
class IamaxConfig:
    """参数对象：封装 Iamax 测试的规模、类型及名称"""
    n: int
    dtype: torch.dtype
    name: str = "Case"


class IamaxTester:
    """Iamax (寻找最大值索引) 算子测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        blocking_env = os.getenv("ASCEND_LAUNCH_BLOCKING", "0")
        mode_str = "异步 (Async)" if blocking_env == "0" else "同步 (Blocking)"
        logger.info("当前 NPU 模式: %s (BLOCKING=%s)", mode_str, blocking_env)

    @staticmethod
    def get_tensor(n: int, dtype: torch.dtype, device: str) -> torch.Tensor:
        """
        静态方法：生成测试 Tensor 并注入唯一最大值。
        """
        # 1. CPU 上生成基础数据
        if dtype == torch.complex64:
            real = torch.randn(n, dtype=torch.float32)
            imag = torch.randn(n, dtype=torch.float32)
            vec = torch.complex(real, imag)
        else:
            vec = torch.randn(n, dtype=torch.float32)

        # 2. 注入极大值以避免精度导致的平局
        max_idx = random.randint(0, n - 1)
        if dtype == torch.complex64:
            vec[max_idx] = complex(10000.0, 10000.0)
        else:
            vec[max_idx] = 10000.0

        return vec.to(device).contiguous()

    def run_case(self, cfg: IamaxConfig) -> bool:
        """
        执行单次测试用例。
        G.ERR.01: 最小化 try 块。
        """
        # 1. 准备数据
        vec_x = self.get_tensor(cfg.n, cfg.dtype, self.device)

        # 2. 核心调用 (G.ERR.01: 仅包裹算子执行及其同步)
        try:
            res_npu = torch_sip.asd_blas_iamax(vec_x)
        except Exception as exc:
            logger.error("[%s] 算子执行崩溃: %s", cfg.name, exc)
            return False

        # 3. CPU 参考计算 (移出 try 块)
        x_cpu = vec_x.cpu()
        npu_idx = res_npu.item() if isinstance(res_npu, torch.Tensor) else res_npu

        if cfg.dtype == torch.float32:
            # Isamax: max(|x_i|)
            ref_idx = torch.argmax(torch.abs(x_cpu)).item()
            op_name = "Isamax"
        else:
            # Icamax: max(|Re(x_i)| + |Im(x_i)|)
            val_comp = torch.abs(x_cpu.real) + torch.abs(x_cpu.imag)
            ref_idx = torch.argmax(val_comp).item()
            op_name = "Icamax"

        # 4. 索引比对逻辑 (拆分 IF 语句)
        is_passed = False
        base_info = "Unknown"

        if npu_idx == ref_idx:
            is_passed = True
            base_info = "0-based"

        if npu_idx == (ref_idx + 1):
            is_passed = True
            base_info = "1-based"

        # 5. 日志输出
        status = "PASS" if is_passed else "FAIL"
        dtype_str = "F32" if cfg.dtype == torch.float32 else "C64"
        logger.info("[%s] %-20s | %s | %s | N=%d",
                    status, cfg.name, op_name, dtype_str, cfg.n)

        if not is_passed:
            logger.error("   NPU Index: %s | CPU Ref: %d (0-based) / %d (1-based)",
                         npu_idx, ref_idx, ref_idx + 1)
        else:
            logger.info("   - Detected BLAS standard: %s", base_info)

        return is_passed


def main():
    """测试主入口"""
    tester = IamaxTester()
    logger.info("开始 Iamax (Isamax/Icamax) 专项测试 ...\n")

    # 使用参数对象定义测试套件
    test_suites = [
        IamaxConfig(1024, torch.float32, name="Float_Small"),
        IamaxConfig(65536, torch.float32, name="Float_Large"),
        IamaxConfig(1024, torch.complex64, name="Complex_Small"),
        IamaxConfig(65536, torch.complex64, name="Complex_Large"),
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