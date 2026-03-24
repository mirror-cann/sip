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
import math
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
logger = logging.getLogger("torch_sip_csrot_test")


@dataclass(frozen=True)
class RotConfig:
    """参数对象：封装旋转测试的规模与角度配置"""
    n: int
    angle_deg: float
    name: str = "Case"


class CsrotTester:
    """复数向量平面旋转 (CSROT) 算子测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        blocking_env = os.getenv("ASCEND_LAUNCH_BLOCKING", "0")
        mode_str = "异步 (Async)" if blocking_env == "0" else "同步 (Blocking)"
        logger.info("当前 NPU 运行模式: %s (ASCEND_LAUNCH_BLOCKING=%s)", mode_str, blocking_env)

    @staticmethod
    def get_complex_tensor(n: int, device: str) -> torch.Tensor:
        """
        静态方法：安全生成复数 Tensor。
        整改：先在 CPU 生成随机数再搬移至目标设备，确保复数随机生成的稳定性。
        """
        real = torch.randn(n, dtype=torch.float32)
        imag = torch.randn(n, dtype=torch.float32)
        return torch.complex(real, imag).to(device).contiguous()

    def run_case(self, cfg: RotConfig) -> bool:
        """
        执行单次测试用例。
        G.ERR.01: 最小化 try 块，仅包裹算子核心调用逻辑。
        """
        # 1. 准备数据与旋转系数
        rad = math.radians(cfg.angle_deg)
        c_val = math.cos(rad)
        s_val = math.sin(rad)

        vec_x = self.get_complex_tensor(cfg.n, self.device)
        vec_y = self.get_complex_tensor(cfg.n, self.device)

        # 备份初始数据用于 CPU 参考计算
        x_init = vec_x.cpu()
        y_init = vec_y.cpu()

        # 2. 核心调用 (G.ERR.01: 仅包裹原地修改的算子执行及其同步)
        try:
            torch_sip.asd_blas_csrot(vec_x, vec_y, c_val, s_val)
        except Exception as exc:
            logger.error("[%s] 算子执行崩溃: %s", cfg.name, exc)
            return False

        # 3. CPU 参考计算 (移出 try 块)
        # 公式：x' = c*x + s*y, y' = -s*x + c*y
        x_ref = c_val * x_init + s_val * y_init
        y_ref = -s_val * x_init + c_val * y_init

        # 4. 精度比对与验证
        vec_x_cpu = vec_x.cpu()
        vec_y_cpu = vec_y.cpu()

        x_close = torch.allclose(vec_x_cpu, x_ref, rtol=1e-4, atol=1e-4)
        y_close = torch.allclose(vec_y_cpu, y_ref, rtol=1e-4, atol=1e-4)
        is_close = x_close and y_close

        status = "PASS" if is_close else "FAIL"
        logger.info("[%s] %-20s | N=%d | Angle=%s° | c=%.3f, s=%.3f",
                    status, cfg.name, cfg.n, cfg.angle_deg, c_val, s_val)

        if not is_close:
            if not x_close:
                logger.error("   [X Diff Max]: %.6e", torch.abs(vec_x_cpu - x_ref).max())
            if not y_close:
                logger.error("   [Y Diff Max]: %.6e", torch.abs(vec_y_cpu - y_ref).max())

        return is_close


def main():
    """测试主套件入口"""
    tester = CsrotTester()
    logger.info("开始 Csrot (复数向量旋转) 专项测试 (参数对象化重构版本) ...\n")

    # 使用参数对象定义测试套件
    test_suites = [
        RotConfig(1024, 30.0, "Rotate_30_Deg"),
        RotConfig(2048, 45.0, "Rotate_45_Deg"),
        RotConfig(2048, 90.0, "Rotate_90_Deg"),
        RotConfig(4096, 180.0, "Rotate_180_Deg"),
        RotConfig(8, -60.0, "Rotate_Negative"),
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