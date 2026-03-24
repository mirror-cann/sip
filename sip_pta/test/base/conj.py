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
import sys

import torch
import torch_npu

# 配置全局日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("torch_sip_test")


def get_torch_sip():
    """导入插件模块，失败则抛出 ImportError"""
    try:
        import torch_sip
        logger.info("✓ torch_sip module loaded successfully")
        return torch_sip
    except ImportError as e:
        error_msg = (
            f"Failed to import torch_sip: {e}\n"
            "Please build extension first:\n"
            "  cd torch_sip && python setup.py build_ext --inplace"
        )
        raise ImportError(error_msg) from e


def initialize_device():
    """初始化 NPU 设备，失败则抛出 RuntimeError"""
    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available. This test requires NPU hardware.")

    device = torch.device("npu:0")
    # 显式设置设备，确保后续算子调度正确
    torch.npu.set_device(device)
    logger.info("✓ NPU available. Using device: %s", device)
    return device


def verify_precision(result, expected):
    """验证精度并输出差异样本"""
    result_cpu = result.cpu()
    expected_cpu = expected.cpu()

    if torch.allclose(result_cpu, expected_cpu, rtol=1e-3, atol=1e-3):
        logger.info("✓ Result matches expected torch.conj(x) on CPU")
        return True

    logger.error("✗ Result does not match expected")
    # 仅在调试或错误时打印数据样本
    logger.error("Result sample: %s", result_cpu.flatten()[:2])
    logger.error("Expect sample: %s", expected_cpu.flatten()[:2])
    return False


def test_conj(torch_sip, device):
    """测试 asd_conj 算子逻辑"""
    logger.info("-" * 40)
    logger.info("Testing asd_conj operation")

    # 构造复数输入
    shape = (4, 4)
    x_real = torch.randn(shape, device=device)
    x_imag = torch.randn(shape, device=device)
    x = torch.complex(x_real, x_imag)

    logger.info("Input tensor: shape=%s, dtype=%s", x.shape, x.dtype)

    # 预期结果
    expected = torch.conj(x)

    try:
        # 执行插件算子
        result = torch_sip.conj(x)
        logger.info("✓ asd_conj executed successfully")
        return verify_precision(result, expected)
    except Exception:
        logger.exception("✗ asd_conj execution failed due to internal error")
        return False


def main():
    """主程序入口"""
    logger.info("torch_sip Extension Test")
    logger.info("=" * 40)
    logger.info("PyTorch version: %s", torch.__version__)
    logger.info("Python version: %s", sys.version.split()[0])

    try:
        # 1. 环境准备
        sip_module = get_torch_sip()
        device = initialize_device()

        # 2. 运行测试
        success = test_conj(sip_module, device)

        # 3. 汇总报告
        logger.info("=" * 40)
        if success:
            logger.info("All tests passed! ✓")
            return 0

        logger.error("Some tests failed. ✗")
        return 1

    except (ImportError, RuntimeError) as e:
        logger.error("Setup failed: %s", e)
        return 1
    except Exception:
        logger.exception("An unhandled exception occurred")
        return 1


if __name__ == "__main__":
    # 仅在脚本最顶层处理进程退出
    sys.exit(main())