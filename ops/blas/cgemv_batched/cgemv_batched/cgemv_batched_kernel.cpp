/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mki/utils/math/tensor_utils.h"
#include "mki/utils/math/math.h"
#include "mki_loader/op_register.h"
#include "utils/assert.h"
#include "log/log.h"
#include "tiling/cgemv_batched_tiling_data.h"
#include "tiling/cgemv_batched_tiling.h"
#include "cgemv_batched.h"

static constexpr int64_t INPUT_NUM = 3;
static constexpr int64_t OUTPUT_NUM = 1;

namespace AsdSip {
class CgemvBatchedKernel : public KernelBase {
public:
    explicit CgemvBatchedKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::CgemvBatched),
            "CgemvBatched: param type invalid", return false);
        ASDSIP_CHECK(launchParam.GetInTensorCount() == INPUT_NUM, "input num invalid", return false);
        ASDSIP_CHECK(launchParam.GetOutTensorCount() == OUTPUT_NUM, "output num invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(CgemvBatchedTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = CgemvBatchedTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitImpl CgemvBatchedTiling failed",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
        return Status::OkStatus();
    }
};

// CgemvBatchedC32C64Kernel
class CgemvBatchedC32C64Kernel : public CgemvBatchedKernel {
public:
    explicit CgemvBatchedC32C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : CgemvBatchedKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(CgemvBatchedKernel::CanSupport(launchParam), "failed to check support", return false);
        return true;
    }
};
REG_KERNEL_BASE(CgemvBatchedC32C64Kernel);
}  // namespace AsdSip