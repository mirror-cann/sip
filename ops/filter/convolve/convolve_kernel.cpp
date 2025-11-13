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
#include "convolve/tiling/convolve_tiling.h"
#include "convolve/tiling/convolve_tiling_data.h"
#include "convolve.h"

namespace AsdSip {
using namespace Mki;
class ConvolveBaseKernel : public KernelBase {
public:
    explicit ConvolveBaseKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Convolve), "Convolve: param type invalid", return false);
        ASDSIP_CHECK(launchParam.GetInTensorCount() == 2, "input num invalid", return false);
        ASDSIP_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);  // markhere
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(ConvolveTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = ConvolveTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl ConvolveTiling failed",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
        kernelInfo_.SetHwsyncIdx(0);
        return Status::OkStatus();
    }
};

// SasumF32Kernel
class ConvolveKernel : public ConvolveBaseKernel {
public:
    explicit ConvolveKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ConvolveBaseKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(ConvolveBaseKernel::CanSupport(launchParam), "failed to check support", return false);
        return true;
    }
};
REG_KERNEL_BASE(ConvolveKernel);
}  // namespace AsdSip