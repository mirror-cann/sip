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
#include "utils/assert.h"
#include "log/log.h"
#include "dot.h"
#include "tiling/sdot_tiling_data.h"
#include "tiling/sdot_tiling.h"

using namespace Mki;

namespace AsdSip {

class SdotKernel : public KernelBase {
public:
    explicit SdotKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Dot), "Sdot: param type invalid", return false);
        ASDSIP_CHECK(launchParam.GetInTensorCount() == 2, "Input num invalid", return false);
        ASDSIP_CHECK(launchParam.GetOutTensorCount() == 1, "Output num invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(SdotTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = SdotTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl DotTiling failed",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
        kernelInfo_.SetHwsyncIdx(0);
        return Status::OkStatus();
    }
};

// DotC64Kernel
class SdotF32Kernel : public SdotKernel {
public:
    explicit SdotF32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : SdotKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(SdotKernel::CanSupport(launchParam), "failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT, "tensor 0 dtype unsupported",
                  return false);
        ASDSIP_CHECK(launchParam.GetInTensor(1).desc.dtype == TENSOR_DTYPE_FLOAT, "tensor 1 dtype unsupported",
                  return false);
        return true;
    }
};
REG_KERNEL_BASE(SdotF32Kernel);
}  // namespace AsdSip