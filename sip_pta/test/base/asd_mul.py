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
    """
    检查并导入 torch_sip 模块。
    """
    try:
        import torch_sip
        logger.info("✓ torch_sip module loaded successfully.")
        return torch_sip
    except ImportError as e:
        error_msg = (
            f"Failed to import torch_sip: {e}\n"
            "Please build extension first:\n"
            "  cd torch_sip && python setup.py build_ext --inplace"
        )
        raise ImportError(error_msg) from e


def initialize_npu():
    """
    初始化 NPU 设备。
    """
    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available. This test requires NPU hardware.")

    device = torch.device('npu:0')
    torch.npu.set_device(device)
    logger.info("✓ NPU initialized. Using device: %s", device)
    return device


def verify_complex_mul(result, expected):
    """
    验证复数乘法结果。
    """
    res_cpu = result.cpu()
    exp_cpu = expected.cpu()

    if torch.allclose(res_cpu, exp_cpu, rtol=1e-3, atol=1e-3):
        logger.info("✓ Precision check passed: result matches expected (x * y).")
        return True

    logger.error("✗ Precision check failed!")
    logger.debug("Result sample: %s", res_cpu.flatten()[:2])
    logger.debug("Expect sample: %s", exp_cpu.flatten()[:2])
    return False


def test_asd_mul(torch_sip, device):
    """
    执行 asd_mul 测试逻辑。
    """
    logger.info("-" * 40)
    logger.info("Testing asd_mul operation...")

    # 构造输入数据
    shape = (4, 4)
    x = torch.complex(torch.randn(shape, device=device), torch.randn(shape, device=device))
    y = torch.complex(torch.randn(shape, device=device), torch.randn(shape, device=device))

    logger.info("Input tensors: shape=%s, dtype=%s", x.shape, x.dtype)

    # 计算预期值与实际值
    expected = x * y
    result = torch_sip.asd_mul(x, y)

    return verify_complex_mul(result, expected)


def main():
    """
    主入口函数。负责异常捕获和最终退出码返回。
    """
    logger.info("Starting torch_sip extension test...")
    logger.info("PyTorch: %s | Python: %s", torch.__version__, sys.version.split()[0])

    try:
        # 1. 环境与设备准备
        sip_module = get_torch_sip()
        npu_device = initialize_npu()

        # 2. 运行测试用例
        success = test_asd_mul(sip_module, npu_device)

        # 3. 汇总结果
        logger.info("=" * 40)
        if success:
            logger.info("Status: ALL TESTS PASSED")
            return 0

        logger.error("Status: TEST FAILED")
        return 1

    except (ImportError, RuntimeError) as e:
        logger.error("Environment Error: %s", e)
        return 1
    except Exception:
        logger.exception("An unexpected fatal error occurred during testing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())