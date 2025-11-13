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
#include "mki/base/kernel_base.h"
#include "utils/assert.h"
#include "log/log.h"
#include "tiling/cal_tiling.h"
#include "cal.h"
#include "common_tiling_data.h"
#include "tiling/cal_tiling_data.h"

static constexpr uint32_t CSCAL_TENSOR_INPUT_NUM = 2;
static constexpr uint32_t SSCAL_TENSOR_INPUT_NUM = 1;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 0;

using namespace Mki;

namespace AsdSip {
class CalKernel : public KernelBase {
public:
    explicit CalKernel(const std::string &kernelName, const BinHandle *handle) noexcept : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        OpParam::Cal param = AnyCast<OpParam::Cal>(launchParam.GetParam());
        if (param.calType == OpParam::Cal::CalType::CAL_CSCAL) {
            ASDSIP_CHECK(launchParam.GetInTensorCount() == CSCAL_TENSOR_INPUT_NUM, "input num invalid", return false);
        } else {
            ASDSIP_CHECK(launchParam.GetInTensorCount() == SSCAL_TENSOR_INPUT_NUM, "input num invalid", return false);
        }
        ASDSIP_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM, "output num invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        uint64_t tilingSize = sizeof(CommonTilingData) + sizeof(CalTilingData);
        OpParam::Cal param = AnyCast<OpParam::Cal>(launchParam.GetParam());
        if (param.calType == OpParam::Cal::CalType::CAL_CSCAL) {
            tilingSize = sizeof(uint32_t) + sizeof(CalTilingData);
        }
        return tilingSize;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = CalTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl CalTiling failed",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
        OpParam::Cal param = AnyCast<OpParam::Cal>(launchParam.GetParam());
        if (param.calType == OpParam::Cal::CalType::CAL_CSCAL) {
            kernelInfo_.SetHwsyncIdx(0);
        }
        return Status::OkStatus();
    }
};

// SscalF32Kernel
class SscalF32Kernel : public CalKernel {
public:
    explicit SscalF32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : CalKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(CalKernel::CanSupport(launchParam), "failed to check support", return false);
        auto opParam = AnyCast<OpParam::Cal>(launchParam.GetParam());
        OpParam::Cal::CalType type = opParam.calType;
        ASDSIP_CHECK(type == OpParam::Cal::CalType::CAL_SSCAL || type == OpParam::Cal::CalType::CAL_CSSCAL, "Sscal: param type invalid",
                  return false);
        return true;
    }
};
REG_KERNEL_BASE(SscalF32Kernel);

// CscalC64Kernel
class CscalC64Kernel : public CalKernel {
public:
    explicit CscalC64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : CalKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(CalKernel::CanSupport(launchParam), "failed to check support", return false);
        auto opParam = AnyCast<OpParam::Cal>(launchParam.GetParam());
        OpParam::Cal::CalType type = opParam.calType;
        ASDSIP_CHECK(type == OpParam::Cal::CalType::CAL_CSCAL, "Cscal: param type invalid", return false);
        return true;
    }
};
REG_KERNEL_BASE(CscalC64Kernel);
}  // namespace AsdSip