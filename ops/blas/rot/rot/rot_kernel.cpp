/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mki_loader/op_register.h"
#include "utils/assert.h"
#include "log/log.h"
#include "tiling/rot_tiling.h"
#include "tiling/rot_tiling_data.h"
#include "rot.h"

static constexpr uint32_t TENSOR_INPUT_NUM = 2;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 0;

using namespace Mki;

namespace AsdSip {
class RotKernel : public KernelBase {
public:
    explicit RotKernel(const std::string &kernelName, const BinHandle *handle) noexcept : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Rot), "Rot: param type invalid", return false);
        ASDSIP_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "input num invalid", return false);
        ASDSIP_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM, "output num invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(RotTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = RotTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl RotTiling failed",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
        OpParam::Rot param = AnyCast<OpParam::Rot>(launchParam.GetParam());
        if (param.rotType == OpParam::Rot::RotType::ROT_CSROT) {
            kernelInfo_.SetHwsyncIdx(0);
        }
        return Status::OkStatus();
    }
};

// CsrotC64Kernel
class CsrotC64Kernel : public RotKernel {
public:
    explicit CsrotC64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : RotKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(RotKernel::CanSupport(launchParam), "failed to check support", return false);
        auto opParam = AnyCast<OpParam::Rot>(launchParam.GetParam());
        OpParam::Rot::RotType type = opParam.rotType;
        ASDSIP_CHECK(type == OpParam::Rot::RotType::ROT_CSROT, "Csrot: param type invalid", return false);
        return true;
    }
};
REG_KERNEL_BASE(CsrotC64Kernel);
}  // namespace AsdSip