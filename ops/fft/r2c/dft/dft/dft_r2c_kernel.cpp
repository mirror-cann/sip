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
#include "utils/assert.h"
#include "log/log.h"
#include "tiling/dft_r2c_tiling.h"
#include "tiling/dft_r2c_tiling_data.h"
#include "dft_r2c.h"

namespace AsdSip {
using namespace Mki;
class DftR2CKernel : public KernelBase {
public:
    explicit DftR2CKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::DftR2C),
            "DftR2C: param type invalid", return false);
        ASDSIP_CHECK(launchParam.GetInTensorCount() == 2, "input num invalid", return false);
        ASDSIP_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);  // markhere
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(DftR2CTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = DftR2CTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl DftR2CTiling failed",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
        return Status::OkStatus();
    }
};

// DftR2CC64Kernel
class DftR2CC64Kernel : public DftR2CKernel {
public:
    explicit DftR2CC64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : DftR2CKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(DftR2CKernel::CanSupport(launchParam), "failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

REG_KERNEL_BASE(DftR2CC64Kernel);
}  // namespace AsdOps