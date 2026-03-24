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
logger = logging.getLogger("torch_sip_nrm2_test")


@dataclass(frozen=True)
class Nrm2Config:
    """参数对象：封装 Nrm2 测试的维度、类型及名称"""
    n: int
    dtype: torch.dtype
    name: str = "Case"


class Nrm2Tester:
    """Nrm2 (向量范数) 算子测试类"""

    def __init__(self, device: str = "npu:0"):
        self.device = device
        blocking_env = os.getenv("ASCEND_LAUNCH_BLOCKING", "0")
        mode_str = "异步 (Async)" if blocking_env == "0" else "同步 (Blocking)"
        logger.info("当前 NPU 模式: %s (BLOCKING=%s)", mode_str, blocking_env)

    @staticmethod
    def get_tensor(n: int, dtype: torch.dtype, device: str) -> torch.Tensor:
        """
        静态方法：安全生成测试 Tensor。
        整改：先在 CPU 生成随机数再搬移至目标设备。
        """
        if dtype == torch.complex64:
            real = torch.randn(n, dtype=torch.float32)
            imag = torch.randn(n, dtype=torch.float32)
            vec = torch.complex(real, imag)
        elif dtype == torch.float32:
            vec = torch.randn(n, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        return vec.to(device).contiguous()

    def run_case(self, cfg: Nrm2Config) -> bool:
        """
        执行单次测试用例。
        G.ERR.01: 最小化 try 块。
        """
        # 1. 准备数据
        vec_x = self.get_tensor(cfg.n, cfg.dtype, self.device)

        # 2. 核心调用 (G.ERR.01: 仅包裹算子执行及其同步)
        try:
            res_npu = torch_sip.asd_blas_nrm2(vec_x)
        except Exception as exc:
            logger.error("[%s] 算子执行崩溃: %s", cfg.name, exc)
            return False

        # 3. CPU 参考计算 (移出 try 块)
        x_cpu = vec_x.cpu()
        npu_val = res_npu.item() if isinstance(res_npu, torch.Tensor) else res_npu
        ref_val = torch.linalg.vector_norm(x_cpu).item()

        # 确定算子名称
        op_name = "Snrm2"
        if cfg.dtype == torch.complex64:
            op_name = "Scnrm2"

        # 4. 精度判定 (拆分逻辑)
        is_close = torch.isclose(
            torch.tensor(npu_val),
            torch.tensor(ref_val),
            rtol=1e-4,
            atol=1e-4
        ).item()

        # 5. 日志输出
        status = "PASS" if is_close else "FAIL"
        dtype_str = "F32" if cfg.dtype == torch.float32 else "C64"
        logger.info("[%s] %-20s | %s | %s | N=%d",
                    status, cfg.name, op_name, dtype_str, cfg.n)

        if not is_close:
            logger.error("   NPU Result: %.6f | CPU Ref: %.6f | Diff: %.6e",
                         npu_val, ref_val, abs(npu_val - ref_val))

        return is_close


def main():
    """测试主入口"""
    tester = Nrm2Tester()
    logger.info("开始 Nrm2 (Snrm2/Scnrm2) 专项测试 ...\n")

    # 使用参数对象定义测试套件
    test_suites = [
        Nrm2Config(1024, torch.float32, name="Float_Small"),
        Nrm2Config(65536, torch.float32, name="Float_Large"),
        Nrm2Config(1024, torch.complex64, name="Complex_Small"),
        Nrm2Config(65536, torch.complex64, name="Complex_Large"),
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