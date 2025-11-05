/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
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

#include "fft_b.h"
#include "tiling/fft_b_tiling.h"
#include "tiling/tiling_data.h"

static constexpr uint32_t TENSOR_INPUT_CNT = 4;
static constexpr uint32_t TENSOR_OUTPUT_CNT = 1;
namespace AsdSip {

using Mki::KernelInfo;
using Mki::LaunchParam;
using Mki::BinHandle;
using Mki::KernelBase;
using Mki::Status;
using Mki::TENSOR_DTYPE_INT32;
using Mki::TENSOR_DTYPE_FLOAT;
using Mki::TENSOR_DTYPE_COMPLEX64;

class FftBKernel : public KernelBase {
public:
    explicit FftBKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_CNT, "check inTensor count failed", return false);
        ASDSIP_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_CNT,
            "check outTensor count failed", return false);

        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FftB), "check param type failed!", return false);
        auto &param = AnyCast<OpParam::FftB>(launchParam.GetParam());
        ASDSIP_CHECK(param.radixVec.size() == 2, "check radixVec failed!", return false);
        auto inTensor0Dtype = launchParam.GetInTensor(0).desc.dtype;
        auto inTensor1Dtype = launchParam.GetInTensor(1).desc.dtype;
        auto inTensor2Dtype = launchParam.GetInTensor(2).desc.dtype;
        auto inTensor3Dtype = launchParam.GetInTensor(3).desc.dtype;
        ASDSIP_CHECK(inTensor0Dtype == TENSOR_DTYPE_COMPLEX64, "inTensor 0 dtype unsupported", return false);
        ASDSIP_CHECK(inTensor1Dtype == TENSOR_DTYPE_FLOAT, "inTensor 1 dtype unsupported", return false);
        ASDSIP_CHECK(inTensor2Dtype == TENSOR_DTYPE_FLOAT, "inTensor 2 dtype unsupported", return false);
        ASDSIP_CHECK(inTensor3Dtype == TENSOR_DTYPE_INT32, "inTensor 3 dtype unsupported", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(FftBTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_910B) {
            kernelInfo_.SetHwsyncIdx(0);
        }
        auto status = FftBTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitKernelInfoImpl FftBTiling failed",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
        return Status::OkStatus();
    }
};

class FftBC64Kernel : public FftBKernel {
public:
    explicit FftBC64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftBKernel(kernelName, handle)
    {
    }
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftBKernel::CanSupport(launchParam), "failed to check support", return false);
        return true;
    }
};

REG_KERNEL_BASE(FftBC64Kernel);

}  // namespace AsdSip