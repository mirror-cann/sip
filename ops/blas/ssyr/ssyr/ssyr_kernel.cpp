/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mki_loader/op_register.h>
#include "utils/assert.h"
#include "log/log.h"
#include "mki/utils/math/tensor_utils.h"
#include "mki/utils/math/math.h"
#include "tiling/ssyr_tiling_data.h"
#include "tiling/ssyr_tiling.h"
#include "ssyr.h"

using namespace Mki;

namespace AsdSip {
static constexpr int64_t TENSOR_INPUT_NUM = 1;
static constexpr int64_t TENSOR_OUTPUT_NUM = 1;

class SsyrKernel : public KernelBase {
public:
    explicit SsyrKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Ssyr), "Ssyr: param type invalid", return false);
        ASDSIP_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "input num invalid", return false);
        ASDSIP_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM, "output num invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(SsyrTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = SsyrTiling(launchParam, kernelInfo_);  // 调用tiling函数
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl SsyrTiling failed",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
        return Status::OkStatus();
    }
};

// SsyrF32Kernel
class SsyrF32Kernel : public SsyrKernel {
public:
    explicit SsyrF32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : SsyrKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(SsyrKernel::CanSupport(launchParam), "failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT, "in tensor dtype unsupported",
                  return false);
        ASDSIP_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT, "out tensor dtype unsupported",
                  return false);
        return true;
    }
};
REG_KERNEL_BASE(SsyrF32Kernel);
}  // namespace AsdSip