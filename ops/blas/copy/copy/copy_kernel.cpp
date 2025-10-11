/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki_loader/op_register.h>
#include "utils/assert.h"
#include "log/log.h"
#include "mki/base/kernel_base.h"
#include "tiling/copy_tiling.h"
#include "common_tiling_data.h"
#include "copy.h"

static constexpr uint32_t TENSOR_INPUT_CNT = 1;
static constexpr uint32_t TENSOR_OUTPUT_CNT = 1;

using namespace Mki;

namespace AsdSip {
class CopyKernel : public KernelBase {
public:
    explicit CopyKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Copy), "Cal: param type invalid", return false);
        ASDSIP_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_CNT, "input num invalid", return false);
        ASDSIP_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_CNT, "output num invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        uint64_t tilingSize = static_cast<uint64_t>(sizeof(CommonTilingData));
        return tilingSize;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = CopyTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl CopyTiling failed",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
        return Status::OkStatus();
    }
};

// CopyF32Kernel
class CopyF32Kernel : public CopyKernel {
public:
    explicit CopyF32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : CopyKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(CopyKernel::CanSupport(launchParam), "failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == launchParam.GetOutTensor(0).desc.dtype,
            "dtypes of input tensor and output tensor are not consistent", return false);

        return true;
    }
};
REG_KERNEL_BASE(CopyF32Kernel);

}  // namespace AsdSip