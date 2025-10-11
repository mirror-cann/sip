/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <mki_loader/op_register.h>
#include "interpbycoeff.h"
#include "mki/utils/math/tensor_utils.h"
#include "mki/utils/math/math.h"
#include "utils/assert.h"
#include "log/log.h"
#include "tiling/interpbycoeff_tiling_data.h"
#include "tiling/interpbycoeff_tiling.h"
#include "tiling/tiling_api.h"

namespace AsdSip {
using namespace Mki;

static constexpr uint32_t TENSOR_INPUT_NUM = 2;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 1;

class InterpByCoeffKernel : public KernelBase {
public:
    explicit InterpByCoeffKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {}

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::InterpByCoeff),
            "Interpolation: param type invalid",
            return false);
        ASDSIP_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "input num invalid", return false);
        ASDSIP_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM, "output num invalid", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64 ||
                         launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX32,
            "tensor dtype unsupported",
            return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        optiling::TCubeTiling tCubeTiling;
        uint32_t tSize = tCubeTiling.GetDataSize();
        return sizeof(InterpByCoeffTilingData) + tSize;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = InterpByCoeffTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS,
            "InitRunInfoImpl AsumTiling failed",
            return Status::FailStatus(ERROR_INVALID_VALUE));
        kernelInfo_.SetHwsyncIdx(0);
        return Status::OkStatus();
    }
};

REG_KERNEL_BASE(InterpByCoeffKernel);
}  // namespace AsdSip