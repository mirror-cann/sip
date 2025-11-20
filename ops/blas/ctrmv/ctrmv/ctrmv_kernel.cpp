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
#include "tiling/ctrmv_tiling.h"
#include "ctrmv.h"
#include "tiling/ctrmv_tiling_data.h"

using namespace Mki;

namespace AsdSip {
static constexpr int64_t TENSOR_INPUT_NUM = 3;
static constexpr int64_t TENSOR_OUTPUT_NUM = 0;

class CtrmvKernel : public KernelBase {
public:
    explicit CtrmvKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Ctrmv),
            "Ctrmv: param type invalid", return false);
        ASDSIP_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "input num invalid", return false);
        ASDSIP_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM, "output num invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(CtrmvTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = CtrmvTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl CtrmvTiling failed",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
        kernelInfo_.SetHwsyncIdx(0);
        return Status::OkStatus();
    }
};

// CtrmvC64Kernel
class CtrmvC64Kernel : public CtrmvKernel {
public:
    explicit CtrmvC64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : CtrmvKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(CtrmvKernel::CanSupport(launchParam), "failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        ASDSIP_CHECK(launchParam.GetInTensor(1).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        return true;
    }
};
REG_KERNEL_BASE(CtrmvC64Kernel);
}  // namespace AsdSip