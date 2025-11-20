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
#include "tiling/sswap_tiling_data.h"
#include "tiling/sswap_tiling.h"
#include "swap.h"

using namespace Mki;

namespace AsdSip {

class SswapKernel : public KernelBase {
public:
    explicit SswapKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Swap), "Sswap: param type invalid", return false);
        ASDSIP_CHECK(launchParam.GetInTensorCount() == 2, "input num invalid", return false);
        ASDSIP_CHECK(launchParam.GetOutTensorCount() == 0, "output num invalid", return false);  // markhere
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(SswapTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = SswapTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl SwapTiling failed",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
        return Status::OkStatus();
    }
};

// SswapF32Kernel
class SswapF32Kernel : public SswapKernel {
public:
    explicit SswapF32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : SswapKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(SswapKernel::CanSupport(launchParam), "failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT, "tensor 0 dtype unsupported",
                  return false);
        ASDSIP_CHECK(launchParam.GetInTensor(1).desc.dtype == TENSOR_DTYPE_FLOAT, "tensor 1 dtype unsupported",
                  return false);
        return true;
    }
};
REG_KERNEL_BASE(SswapF32Kernel);
}  // namespace AsdSip