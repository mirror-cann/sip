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
logger = logging.getLogger("torch_sip_sinc_interp")


@dataclass(frozen=True)
class SincConfig:
    """参数对象：封装 Sinc 插值算子的维度、插值数及量化配置"""
    batch: int
    signal_length: int
    interp_length: int
    interp_num: int = 16
    quant_num: int = 32
    name: str = "Case"


class SincInterpolationTester:
    """Sinc 函数内插值算子功能与边界测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        blocking_env = os.getenv("ASCEND_LAUNCH_BLOCKING", "0")
        mode_str = "异步 (Async)" if blocking_env == "0" else "同步 (Blocking)"
        logger.info("当前 NPU 模式: %s (BLOCKING=%s)", mode_str, blocking_env)

    @staticmethod
    def calculate_tab_size(interp_num: int, quant_num: int) -> int:
        """静态工具方法：根据文档公式计算 sincTab 内部元素总数"""
        return ((quant_num + 1) * 2) * (interp_num * 2 + 8)

    def run_execution_case(self, cfg: SincConfig) -> bool:
        """
        验证算子连通性与输出形状。
        G.ERR.01: 最小化 try 块。
        """
        # 1. 数据准备 (Complex64 输入)
        real = torch.randn((cfg.batch, cfg.signal_length), dtype=torch.float32)
        imag = torch.randn((cfg.batch, cfg.signal_length), dtype=torch.float32)
        input_tensor = torch.complex(real, imag).to(self.device).contiguous()

        # 2. 构造 sincTab [4, N]
        tab_inner_size = self.calculate_tab_size(cfg.interp_num, cfg.quant_num)
        sinc_tab = torch.randn((4, tab_inner_size), dtype=torch.float32, device=self.device).contiguous()

        # 3. 构造索引张量
        seq_floor = torch.arange(cfg.interp_length, dtype=torch.int32) % cfg.signal_length
        pos_floor = seq_floor.unsqueeze(0).expand(cfg.batch, -1).to(self.device).contiguous()

        seq_idx = torch.arange(cfg.interp_length, dtype=torch.int32) % (cfg.quant_num + 1)
        pos_to_tab_index = seq_idx.to(torch.int16).unsqueeze(0).expand(cfg.batch, -1).to(self.device).contiguous()

        # 4. 核心执行 (G.ERR.01: 仅包裹算子核心逻辑)
        try:
            output = torch_sip.rs_interpolation_by_sinc(
                input_tensor, sinc_tab, pos_floor, pos_to_tab_index, cfg.interp_num, cfg.quant_num
            )
        except Exception as exc:
            logger.error("[%s] 运行时崩溃: %s", cfg.name, exc)
            return False

        # 5. 结果校验
        expected_shape = (cfg.batch, cfg.interp_length)
        if output.shape == expected_shape:
            logger.info("[PASS] %-20s | 输出形状正确: %s", cfg.name, list(output.shape))
            return True

        logger.error("[FAIL] %-20s | 形状不匹配! 预期 %s, 实际 %s",
                     cfg.name, expected_shape, output.shape)
        return False

    def test_exception_cases(self) -> bool:
        """验证 PTA 适配层对非法参数的拦截逻辑"""
        logger.info("\n" + "="*50 + "\n开始测试异常边界拦截\n" + "="*50)
        suite_passed = True
        batch, sig_len = 1, 64
        # 基础合法张量
        real = torch.randn((batch, sig_len), dtype=torch.float32)
        imag = torch.randn((batch, sig_len), dtype=torch.float32)
        inp = torch.complex(real, imag).to(self.device).contiguous()
        pos_f = torch.zeros((1, 64), dtype=torch.int32, device=self.device)
        pos_t = torch.zeros((1, 64), dtype=torch.int16, device=self.device)

        # 异常场景定义 (interpNum, quantNum, 描述)
        error_suites = [
            (17, 32, "interpNum > 16 拦截"),
            (15, 32, "interpNum 奇数拦截"),
            (16, 64, "quantNum > 32 拦截"),
            (16, 30, "quantNum 非2的幂拦截"),
        ]

        for i_num, q_num, msg in error_suites:
            tab_size = self.calculate_tab_size(i_num, q_num)
            sinc_tab = torch.randn((4, tab_size), dtype=torch.float32, device=self.device)
            try:
                # G.ERR.01: try 块内只包含核心算子调用
                torch_sip.rs_interpolation_by_sinc(inp, sinc_tab, pos_f, pos_t, i_num, q_num)
            except RuntimeError as exc:
                # 预期内的拦截成功
                err_msg = str(exc).splitlines()[0]
                logger.info("[PASS] %s 成功拦截 -> %s", msg, err_msg)
            else:
                # 如果没有抛出异常，说明拦截失败
                logger.error("[FAIL] %s 失败! 算子透传了非法参数。", msg)
                suite_passed = False

        return suite_passed


def main():
    """测试套件入口"""
    tester = SincInterpolationTester()
    all_passed = True  # 状态跟踪

    # 1. 连通性测试
    test_suites = [
        SincConfig(1, 64, 64, 16, 32, "Normal_Single_Batch"),
        SincConfig(4, 256, 512, 8, 16, "Batched_Upsampling"),
        SincConfig(2, 1024, 256, 12, 32, "Downsampling_Mode"),
    ]

    for config in test_suites:
        # 如果任一用例返回 False，整体状态设为失败
        if not tester.run_execution_case(config):
            all_passed = False

    # 2. 拦截逻辑测试 (由于该方法内部是 print，需要修改其返回值)
    if not tester.test_exception_cases():
        all_passed = False

    # 最终根据结果返回状态码
    if all_passed:
        logger.info("="*50 + "\n所有测试用例已通过 [SUCCESS]\n" + "="*50)
        return 0
    else:
        logger.error("="*50 + "\n部分测试用例未通过 [FAILED]\n" + "="*50)
        return 1


if __name__ == "__main__":
    sys.exit(main())