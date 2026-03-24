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
from typing import Tuple

import torch
import torch_npu
import torch_sip

# 配置全局日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("torch_sip_copy_test")


class CopyTester:
    """BLAS Copy 算子专项测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        # 环境变量读取增加默认值处理
        blocking_env = os.getenv("ASCEND_LAUNCH_BLOCKING", "0")
        mode_str = "异步 (Async)" if blocking_env == "0" else "同步 (Blocking)"
        logger.info("当前 NPU 运行模式: %s (ASCEND_LAUNCH_BLOCKING=%s)", mode_str, blocking_env)

    def get_float_tensor(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """生成 Float32 张量"""
        return torch.randn(shape, dtype=torch.float32, device=self.device)

    def get_complex_tensor(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """生成 Complex64 张量 (采用安全初始化逻辑)"""
        real = torch.randn(shape, dtype=torch.float32)
        imag = torch.randn(shape, dtype=torch.float32)
        return torch.complex(real, imag).to(self.device)

    def run_case_standard(self, shape: Tuple[int, ...], dtype: torch.dtype, name: str = "Standard") -> bool:
        """测试不带 out 参数的版本"""
        # 1. 数据准备
        x_input = self.get_float_tensor(shape) if dtype == torch.float32 else self.get_complex_tensor(shape)
        x_input = x_input.contiguous()

        # 2. 核心调用 (G.ERR.01: 仅包裹算子执行)
        try:
            out_npu = torch_sip.asd_blas_copy(x_input)
        except Exception as exc:
            logger.error("[%s] 算子执行崩溃: %s", name, exc)
            return False

        # 3. 结果验证
        ref = x_input.cpu()
        is_close = torch.equal(out_npu.cpu(), ref)

        status = "PASS" if is_close else "FAIL"
        logger.info("[%s] %-25s | Shape=%-10s | Dtype=%s", status, name, str(shape), dtype)
        return is_close

    def run_case_out(self, shape: Tuple[int, ...], dtype: torch.dtype, name: str = "With Out Param") -> bool:
        """测试带 out 参数的原地覆盖版本"""
        # 1. 数据准备
        vec_x = self.get_float_tensor(shape) if dtype == torch.float32 else self.get_complex_tensor(shape)
        out_tensor = torch.zeros_like(vec_x)

        x_input = vec_x.contiguous()
        out_input = out_tensor.contiguous()

        # 2. 核心调用 (G.ERR.01)
        try:
            # 期望 result_tensor 与 out_input 共享内存
            result_tensor = torch_sip.asd_blas_copy(x_input, out=out_input)
        except Exception as exc:
            logger.error("[%s] 算子执行崩溃: %s", name, exc)
            return False

        # 3. 内存地址检查
        is_same_memory = (result_tensor.data_ptr() == out_input.data_ptr())
        if not is_same_memory:
            logger.error("[%s] 失败: 返回的 Tensor 地址与 out 参数不匹配！", name)
            return False

        # 4. 数值检查
        ref = x_input.cpu()
        is_close = torch.equal(out_input.cpu(), ref)

        status = "PASS" if is_close else "FAIL"
        logger.info("[%s] %-25s | Shape=%-10s | Dtype=%s", status, name, str(shape), dtype)
        return is_close


def main():
    """测试主入口"""
    tester = CopyTester()
    logger.info("开始 BLAS Copy 专项测试 ...\n")

    cases = [
        ((10,), torch.float32, "Float32 1D"),
        ((64, 64), torch.complex64, "Complex64 2D"),
    ]

    all_passed = True
    for shape, dtype, name in cases:
        # 运行常规测试
        if not tester.run_case_standard(shape, dtype, f"{name} (No Out)"):
            all_passed = False

        # 运行带 out 参数测试
        if not tester.run_case_out(shape, dtype, f"{name} (With Out)"):
            all_passed = False

    logger.info("-" * 50)
    if all_passed:
        logger.info("测试结论: ✅ 全部通过")
    else:
        logger.error("测试结论: ❌ 存在失败")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
