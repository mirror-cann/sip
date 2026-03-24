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
logger = logging.getLogger("torch_sip_swap_test")


@dataclass(frozen=True)
class SwapConfig:
    """参数对象：封装向量交换的维度、类型及名称"""
    n: int
    dtype: torch.dtype
    name: str = "Case"


class SwapTester:
    """向量交换 (Swap) 算子测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        blocking_env = os.getenv("ASCEND_LAUNCH_BLOCKING", "0")
        mode_str = "异步 (Async)" if blocking_env == "0" else "同步 (Blocking)"
        logger.info("当前 NPU 运行模式: %s (ASCEND_LAUNCH_BLOCKING=%s)", mode_str, blocking_env)

    @staticmethod
    def get_test_data(n: int, dtype: torch.dtype, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        静态方法：安全生成测试向量。
        整改：先在 CPU 生成随机数再搬移至目标设备。
        """
        if dtype == torch.complex64:
            real_x = torch.randn(n, dtype=torch.float32)
            imag_x = torch.randn(n, dtype=torch.float32)
            vec_x = torch.complex(real_x, imag_x)

            real_y = torch.randn(n, dtype=torch.float32)
            imag_y = torch.randn(n, dtype=torch.float32)
            vec_y = torch.complex(real_y, imag_y)
        else:
            vec_x = torch.randn(n, dtype=dtype)
            vec_y = torch.randn(n, dtype=dtype)

        return vec_x.to(device), vec_y.to(device)

    def run_case(self, cfg: SwapConfig) -> bool:
        """
        执行单次测试用例。
        G.ERR.01: 最小化 try 块。
        """
        # 1. 准备数据
        vec_x, vec_y = self.get_test_data(cfg.n, cfg.dtype, self.device)

        # 2. 备份 CPU 参考结果 (交叉引用)
        ref_x = vec_y.cpu().clone()
        ref_y = vec_x.cpu().clone()

        # 3. 核心调用 (G.ERR.01: 仅包裹原地修改的算子执行)
        try:
            torch_npu.npu.synchronize()
            torch_sip.asd_blas_swap(vec_x, vec_y)
            torch_npu.npu.synchronize()
        except Exception as exc:
            logger.error("[%s] 算子执行崩溃: %s", cfg.name, exc)
            return False

        # 4. 精度验证
        npu_x = vec_x.cpu()
        npu_y = vec_y.cpu()

        # 纯粹的数据搬运，理论上需完全一致（atol=0）
        is_close_x = torch.allclose(npu_x, ref_x, rtol=0, atol=0)
        is_close_y = torch.allclose(npu_y, ref_y, rtol=0, atol=0)
        is_passed = is_close_x and is_close_y

        # 5. 日志输出
        status = "PASS" if is_passed else "FAIL"
        dtype_str = "FP32"
        if cfg.dtype == torch.complex64:
            dtype_str = "C64"

        logger.info("[%s] %-20s | N=%-6d | Dtype=%s",
                    status, cfg.name, cfg.n, dtype_str)

        if not is_passed:
            logger.error("      Data mismatch detected during SWAP.")

        return is_passed


def main():
    """测试主套件入口"""
    tester = SwapTester()
    logger.info("开始向量交换 (SSWAP/CSWAP) 专项测试 ...\n")

    # 使用参数对象定义测试套件
    test_suites = [
        SwapConfig(8, torch.float32, "Sswap_Small"),
        SwapConfig(1024, torch.float32, "Sswap_Medium"),
        SwapConfig(8192, torch.float32, "Sswap_Large"),
        SwapConfig(8, torch.complex64, "Cswap_Small"),
        SwapConfig(1024, torch.complex64, "Cswap_Medium"),
        SwapConfig(8192, torch.complex64, "Cswap_Large"),
    ]

    all_passed = True
    for config in test_suites:
        if not tester.run_case(config):
            all_passed = False

    logger.info("-" * 50)
    if all_passed:
        logger.info("测试结论: ✅ 全部通过")
    else:
        logger.error("测试结论: ❌ 存在失败项")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())