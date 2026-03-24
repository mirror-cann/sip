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
logger = logging.getLogger("torch_sip_ssyr2_test")


@dataclass(frozen=True)
class Sy2Config:
    """参数对象：封装 Ssyr2 测试的维度、标量及三角模式配置"""
    n: int
    alpha: float
    uplo: str
    name: str = "Case"


class Ssyr2Tester:
    """Ssyr2 (实数对称矩阵秩-2 更新) 算子测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        blocking_env = os.getenv("ASCEND_LAUNCH_BLOCKING", "0")
        mode_str = "异步 (Async)" if blocking_env == "0" else "同步 (Blocking)"
        logger.info("当前 NPU 运行模式: %s (ASCEND_LAUNCH_BLOCKING=%s)", mode_str, blocking_env)

    @staticmethod
    def get_tensors(n: int, device: str):
        """静态方法：安全生成测试数据 (先 CPU 后搬移)"""
        mat_a = torch.randn(n, n, dtype=torch.float32)
        vec_x = torch.randn(n, dtype=torch.float32)
        vec_y = torch.randn(n, dtype=torch.float32)
        return mat_a.to(device), vec_x.to(device), vec_y.to(device)

    def run_case(self, cfg: Sy2Config) -> bool:
        """
        执行单次测试用例。
        G.ERR.01: 最小化 try 块，仅包裹算子核心逻辑。
        """
        # 1. 准备数据
        mat_a_npu, vec_x_npu, vec_y_npu = self.get_tensors(cfg.n, self.device)

        # 备份初始数据用于 CPU 参考计算
        mat_a_init_cpu = mat_a_npu.cpu()
        vec_x_cpu = vec_x_npu.cpu()
        vec_y_cpu = vec_y_npu.cpu()

        # 确保输入连续
        mat_a_input = mat_a_npu.clone()
        vec_x_input = vec_x_npu.contiguous()
        vec_y_input = vec_y_npu.contiguous()

        # 2. 核心调用 (G.ERR.01: 仅包裹原地修改的算子执行及其同步)
        try:
            torch_sip.asd_blas_ssyr2(mat_a_input, vec_x_input, vec_y_input, cfg.alpha, cfg.uplo)
        except Exception as exc:
            logger.error("[%s] 算子执行崩溃: %s", cfg.name, exc)
            return False

        # 3. CPU 参考计算 (移出 try 块)
        # 公式：A = alpha * (x * y^T + y * x^T) + A
        update_matrix = cfg.alpha * (torch.outer(vec_x_cpu, vec_y_cpu) +
                                     torch.outer(vec_y_cpu, vec_x_cpu))
        expected_full = mat_a_init_cpu + update_matrix
        a_res_cpu = mat_a_input.cpu()

        # 4. 根据 uplo 拆分三角区域判定 (逻辑拆分)
        is_lower = cfg.uplo.upper() == "L"

        if is_lower:
            # 仅比对下三角
            expected = torch.tril(expected_full)
            result = torch.tril(a_res_cpu)
        else:
            # 仅比对上三角
            expected = torch.triu(expected_full)
            result = torch.triu(a_res_cpu)

        # 5. 验证结果
        is_close = torch.allclose(result, expected, rtol=1e-4, atol=1e-4)

        status = "PASS" if is_close else "FAIL"
        logger.info("[%s] %-20s | N=%d | Uplo=%s | Alpha=%.2f",
                    status, cfg.name, cfg.n, cfg.uplo, cfg.alpha)

        if not is_close:
            max_diff = torch.abs(result - expected).max()
            logger.error("      Max Diff: %.6e", max_diff)

        return is_close


def main():
    """测试主套件入口"""
    tester = Ssyr2Tester()
    logger.info("开始 Ssyr2 (实数对称矩阵秩-2 更新) 专项测试 (参数对象化重构) ...\n")

    # 使用参数对象定义测试套件
    test_suites = [
        Sy2Config(16, 1.0, "L", name="Small_Lower_Update"),
        Sy2Config(32, -0.5, "U", name="Upper_Negative_Alpha"),
        Sy2Config(128, 2.0, "L", name="Large_Lower_Update"),
        Sy2Config(512, 0.1, "U", name="Large_Upper_Update"),
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