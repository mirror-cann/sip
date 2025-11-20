/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki/utils/math/tensor_utils.h>
#include <mki/utils/math/math.h>
#include <mki_loader/op_register.h>
#include <mki/utils/platform/platform_info.h>
#include "utils/assert.h"
#include "log/log.h"

#include "fft_n.h"
#include "tiling/fft_n_tiling.h"
#include "tiling/tiling_data.h"

static constexpr uint32_t TENSOR_INPUT_NUM = 4;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 1;
namespace AsdSip {

using Mki::Status;
using Mki::KernelBase;
using Mki::BinHandle;
using Mki::LaunchParam;
using Mki::KernelInfo;
using Mki::TENSOR_DTYPE_COMPLEX64;
using Mki::TENSOR_DTYPE_FLOAT;
using Mki::TENSOR_DTYPE_INT32;

class FftNKernel : public KernelBase {
public:
    explicit FftNKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "check inTensor count failed", return false);
        ASDSIP_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM,
                "check outTensor count failed", return false);
        auto inTensor0Dtype = launchParam.GetInTensor(0).desc.dtype;
        ASDSIP_CHECK(inTensor0Dtype == TENSOR_DTYPE_COMPLEX64, "inTensor 0 dtype unsupported", return false);
        auto inTensor1Dtype = launchParam.GetInTensor(1).desc.dtype;
        ASDSIP_CHECK(inTensor1Dtype == TENSOR_DTYPE_FLOAT, "inTensor 1 dtype unsupported", return false);
        auto inTensor2Dtype = launchParam.GetInTensor(2).desc.dtype;
        ASDSIP_CHECK(inTensor2Dtype == TENSOR_DTYPE_FLOAT, "inTensor 2 dtype unsupported", return false);
        auto inTensor3Dtype = launchParam.GetInTensor(3).desc.dtype;
        ASDSIP_CHECK(inTensor3Dtype == TENSOR_DTYPE_INT32, "inTensor 3 dtype unsupported", return false);
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FftN), "check param type failed!", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(FftNTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_910B) {
            kernelInfo_.SetHwsyncIdx(0);
        }
        auto status = FftNTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitKernelInfoImpl FftNTiling failed",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
        return Status::OkStatus();
    }
};

class FftNC64Kernel : public FftNKernel {
public:
    explicit FftNC64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftNKernel(kernelName, handle)
    {
    }
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftNKernel::CanSupport(launchParam), "failed to check support", return false);
        return true;
    }
};

REG_KERNEL_BASE(FftNC64Kernel);

}  // namespace AsdSip