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
#include "tiling/caxpy_tiling_data.h"
#include "tiling/caxpy_tiling.h"
#include "caxpy.h"

using namespace Mki;

namespace AsdSip {

class CaxpyKernel : public KernelBase {
public:
    explicit CaxpyKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Caxpy),
            "Caxpy: param type invalid", return false);
        ASDSIP_CHECK(launchParam.GetInTensorCount() == 2, "input num invalid", return false);
        ASDSIP_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);  // markhere

        return true;
    }  // 校验入参出参

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(CaxpyTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = CaxpyTiling(launchParam, kernelInfo_);  // 调用tiling函数
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl CaxpyTiling failed",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
        return Status::OkStatus();
    }
};

// CaxpyC64Kernel
class CaxpyC64Kernel : public CaxpyKernel {
public:
    explicit CaxpyC64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : CaxpyKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(CaxpyKernel::CanSupport(launchParam), "failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        ASDSIP_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        return true;
    }
};
REG_KERNEL_BASE(CaxpyC64Kernel);
}  // namespace AsdSip