/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include <mki/utils/platform/platform_info.h>
#include "utils/assert.h"
#include "log/log.h"
#include "fft_all_mix.h"
#include "fft_c2c_arch35.h"
#include "../../../common/include/tiling/fft_all_mix_tiling_data.h"
#include "tiling/fft_c2c_arch35_tiling.h"

namespace AsdSip {
constexpr uint64_t TENSOR_PERM_IDX = 1;

class FftC2CArch35Kernel : public KernelBase {
public:
    explicit FftC2CArch35Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
            ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FftC2CArch35), "FftC2CArch35: param type invalid",
                return false);
            ASDSIP_CHECK(launchParam.GetInTensorCount() == 4, "input num invalid", return false);
            ASDSIP_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        }
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        uint64_t c2cTilingSize = static_cast<uint64_t>(sizeof(FftAllMixTilingData));
        return c2cTilingSize;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = FftC2CArch35Tiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl FftC2CArch35Tiling failed",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        return Status::OkStatus();
    }
};

class FftC2CArch35C64Kernel : public FftC2CArch35Kernel {
public:
    explicit FftC2CArch35C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftC2CArch35Kernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftC2CArch35Kernel::CanSupport(launchParam), "FftC2CArch35C64Kernel failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

REG_KERNEL_BASE(FftC2CArch35C64Kernel);
}  // namespace AsdSip
