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
#include "tiling/hcgemm_batched_tiling.h"
#include "tiling/hcgemm_batched_tiling_data.h"
#include "hcgemm_batched.h"

namespace AsdSip {
using namespace Mki;
class HCgemmBatchedBaseKernel : public KernelBase {
public:
    explicit HCgemmBatchedBaseKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::HCgemmBatched), "HCgemmBatched: param type invalid", return false);
        ASDSIP_CHECK(launchParam.GetInTensorCount() == 3, "input num invalid", return false);
        ASDSIP_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);  // markhere
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(HCgemmBatchedTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = HCgemmBatchedTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl HCgemmBatchedTiling failed",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
        kernelInfo_.SetHwsyncIdx(0);
        return Status::OkStatus();
    }
};

class HCgemmBatchedC32Kernel : public HCgemmBatchedBaseKernel {
public:
    explicit HCgemmBatchedC32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : HCgemmBatchedBaseKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(HCgemmBatchedBaseKernel::CanSupport(launchParam), "failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX32, "tensor dtype unsupported",
                  return false);
        ASDSIP_CHECK(launchParam.GetInTensor(1).desc.dtype == TENSOR_DTYPE_COMPLEX32, "tensor dtype unsupported",
                  return false);
        ASDSIP_CHECK(launchParam.GetInTensor(2).desc.dtype == TENSOR_DTYPE_UINT32, "tensor dtype unsupported",
                  return false);
        return true;
    }
};
REG_KERNEL_BASE(HCgemmBatchedC32Kernel);
}  // namespace AsdSip