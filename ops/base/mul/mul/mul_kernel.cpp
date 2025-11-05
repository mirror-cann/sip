/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mki/base/kernel_base.h"
#include "mki_loader/op_register.h"
#include "mki/utils/log/log.h"
#include "mki/utils/assert/assert.h"
#include "mul.h"
#include "tiling/mul_tiling.h"
#include "tiling/mul_tiling_data.h"

static constexpr uint32_t TENSOR_INPUT_NUM = 2;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 1;
namespace AsdSip {
using namespace Mki;
class MulKernel : public KernelBase {
public:
    explicit MulKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "check inTensor count failed", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM, "check outTensor count failed", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::CMul), "check param type failed!", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(CMulTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = CMulTiling(launchParam, kernelInfo_);
        MKI_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl CMulTiling failed",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
        return Status::OkStatus();
    }
};

// MulC64Kernel
class MulC64Kernel : public MulKernel {
public:
    explicit MulC64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : MulKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(MulKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64,
            "tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(MulC64Kernel);


// MulC32Kernel
class MulC32Kernel : public MulKernel {
public:
    explicit MulC32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : MulKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(MulKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX32,
            "tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(MulC32Kernel);

}  // namespace AsdSip