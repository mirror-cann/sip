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

import torch
import torch_npu
import torch_sip

# 配置全局日志 (G.LOG.02)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("torch_sip_ssyr_test")


@dataclass(frozen=True)
class SyConfig:
    """参数对象：封装 Ssyr 测试的维度、标量及三角模式配置"""
    n: int
    alpha: float
    uplo: str
    name: str = "Case"


class SsyrTester:
    """Ssyr (对称矩阵秩-1更新) 算子测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        blocking_env = os.getenv("ASCEND_LAUNCH_BLOCKING", "0")
        mode_str = "异步 (Async)" if blocking_env == "0" else "同步 (Blocking)"
        logger.info("当前 NPU 模式: %s (BLOCKING=%s)", mode_str, blocking_env)

    def run_case(self, cfg: SyConfig) -> bool:
        """
        执行单次测试用例。
        G.ERR.01: 最小化 try 块。
        """
        # 1. 准备数据 (安全初始化：先 CPU 后搬移)
        mat_a_init_cpu = torch.randn(cfg.n, cfg.n, dtype=torch.float32)
        vec_x_cpu = torch.randn(cfg.n, dtype=torch.float32)

        mat_a_npu = mat_a_init_cpu.to(self.device)
        vec_x_npu = vec_x_cpu.to(self.device).contiguous()

        # 2. 核心调用 (G.ERR.01: 仅包裹原地修改的算子执行及其同步)
        try:
            torch_sip.asd_blas_ssyr(mat_a_npu, vec_x_npu, cfg.alpha, cfg.uplo)
        except Exception as exc:
            logger.error("[%s] 算子执行崩溃: %s", cfg.name, exc)
            return False

        # 3. CPU 参考计算 (移出 try 块)
        # 核心公式: A = alpha * x * x^T + A
        update_mat = cfg.alpha * torch.outer(vec_x_cpu, vec_x_cpu)
        expected_full = mat_a_init_cpu + update_mat

        a_res_cpu = mat_a_npu.cpu()

        # 4. 根据 uplo 拆分三角区域判定 (逻辑拆分)
        is_lower = cfg.uplo.upper() == "L"

        if is_lower:
            # 仅校验下三角
            expected = torch.tril(expected_full)
            result = torch.tril(a_res_cpu)
        else:
            # 仅校验上三角
            expected = torch.triu(expected_full)
            result = torch.triu(a_res_cpu)

        # 5. 验证结果
        is_close = torch.allclose(result, expected, rtol=1e-4, atol=1e-4)

        status = "PASS" if is_close else "FAIL"
        logger.info("[%s] %-20s | N=%d | Uplo=%s | Alpha=%.2f",
                    status, cfg.name, cfg.n, cfg.uplo, cfg.alpha)

        if not is_close:
            logger.error("      Max Diff: %.6e", torch.abs(result - expected).max())

        return is_close


def main():
    """测试主套件入口"""
    tester = SsyrTester()
    logger.info("开始 Ssyr 专项测试 (参数对象重构版本) ...\n")

    # 使用参数对象定义测试套件
    test_suites = [
        SyConfig(16, 1.0, "L", name="Small_Lower_Update"),
        SyConfig(32, -0.5, "U", name="Upper_Neg_Alpha"),
        SyConfig(128, 2.0, "L", name="Large_Lower_Update"),
        SyConfig(512, 0.1, "U", name="Large_Upper_Update"),
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