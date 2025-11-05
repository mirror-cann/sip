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
#include "tiling/cgemm_tiling_data.h"
#include "tiling/cgemm_tiling.h"
#include "cgemm.h"

constexpr uint8_t INPUT_NUM = 10;
constexpr uint8_t OUTPUT_NUM = 1;

using namespace Mki;

namespace AsdSip {

class CgemmKernel : public KernelBase {
public:
    explicit CgemmKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Cgemm),
            "Cgemm: param type invalid", return false);
        ASDSIP_CHECK(launchParam.GetInTensorCount() == INPUT_NUM, "input num invalid", return false);
        ASDSIP_CHECK(launchParam.GetOutTensorCount() == OUTPUT_NUM, "output num invalid", return false);  // markhere

        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(CgemmTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = CgemmTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl CgemmTiling failed",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
        kernelInfo_.SetHwsyncIdx(0);
        return Status::OkStatus();
    }
};

// CgemmC64Kernel
class CgemmC64Kernel : public CgemmKernel {
public:
    explicit CgemmC64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : CgemmKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(CgemmKernel::CanSupport(launchParam), "failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        ASDSIP_CHECK(launchParam.GetInTensor(1).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        return true;
    }
};
REG_KERNEL_BASE(CgemmC64Kernel);
}  // namespace AsdSip