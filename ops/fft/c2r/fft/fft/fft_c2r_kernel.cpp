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
#include <mki/utils/platform/platform_info.h>
#include "utils/assert.h"
#include "log/log.h"
#include "fft_all_mix.h"
#include "fft_c2r_arch35.h"
#include "../../../common/include/tiling/fft_all_mix_tiling_data.h"
#include "tiling/fft_c2r_tiling.h"

namespace AsdSip {
constexpr uint64_t TENSOR_PERM_IDX = 1;

class FftC2RKernel : public KernelBase {
public:
    explicit FftC2RKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
            ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FftC2RArch35), "FftC2R: param type invalid",
                return false);
            ASDSIP_CHECK(launchParam.GetInTensorCount() == 4, "input num invalid", return false);
            ASDSIP_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        } else if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_910B) {
            ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FftAllMix), "FftC2R: param type invalid",
                return false);
            ASDSIP_CHECK(launchParam.GetInTensorCount() == 8, "input num invalid", return false);
            ASDSIP_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        }

        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        uint64_t c2rTilingSize = static_cast<uint64_t>(sizeof(FftAllMixTilingData));
        return c2rTilingSize;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = FftC2RTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl FftC2RTiling failed",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_910B) {
            kernelInfo_.SetHwsyncIdx(0);
        } 
        
        return Status::OkStatus();
    }
};

// FftC2R1C64Kernel
class FftC2R1C64Kernel : public FftC2RKernel {
public:
    explicit FftC2R1C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftC2RKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftC2RKernel::CanSupport(launchParam), "FftC2R1C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

// FftC2R2C64Kernel
class FftC2R2C64Kernel : public FftC2RKernel {
public:
    explicit FftC2R2C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftC2RKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftC2RKernel::CanSupport(launchParam), "FftC2R2C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

// FftC2R3C64Kernel
class FftC2R3C64Kernel : public FftC2RKernel {
public:
    explicit FftC2R3C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftC2RKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftC2RKernel::CanSupport(launchParam), "FftC2R3C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

// FftC2ROdd1C64Kernel
class FftC2ROdd1C64Kernel : public FftC2RKernel {
public:
    explicit FftC2ROdd1C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftC2RKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftC2RKernel::CanSupport(launchParam), "FftC2ROdd1C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

// FftC2ROdd2C64Kernel
class FftC2ROdd2C64Kernel : public FftC2RKernel {
public:
    explicit FftC2ROdd2C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftC2RKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftC2RKernel::CanSupport(launchParam), "FftC2ROdd2C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

// FftC2ROdd3C64Kernel
class FftC2ROdd3C64Kernel : public FftC2RKernel {
public:
    explicit FftC2ROdd3C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftC2RKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftC2RKernel::CanSupport(launchParam), "FftC2ROdd3C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

// FftC2REven1C64Kernel
class FftC2REven1C64Kernel : public FftC2RKernel {
public:
    explicit FftC2REven1C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftC2RKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftC2RKernel::CanSupport(launchParam), "FftC2REven1C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

// FftC2REven2C64Kernel
class FftC2REven2C64Kernel : public FftC2RKernel {
public:
    explicit FftC2REven2C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftC2RKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftC2RKernel::CanSupport(launchParam), "FftC2REven2C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

// FftC2REven3C64Kernel
class FftC2REven3C64Kernel : public FftC2RKernel {
public:
    explicit FftC2REven3C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftC2RKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftC2RKernel::CanSupport(launchParam), "FftC2REven3C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

// arch35 
// FftC2RC64Kernel
class FftC2RC64Kernel : public FftC2RKernel {
public:
    explicit FftC2RC64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftC2RKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftC2RKernel::CanSupport(launchParam), "FftC2RC64Kernel failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

REG_KERNEL_BASE(FftC2R1C64Kernel);
REG_KERNEL_BASE(FftC2R2C64Kernel);
REG_KERNEL_BASE(FftC2R3C64Kernel);
REG_KERNEL_BASE(FftC2ROdd1C64Kernel);
REG_KERNEL_BASE(FftC2ROdd2C64Kernel);
REG_KERNEL_BASE(FftC2ROdd3C64Kernel);
REG_KERNEL_BASE(FftC2REven1C64Kernel);
REG_KERNEL_BASE(FftC2REven2C64Kernel);
REG_KERNEL_BASE(FftC2REven3C64Kernel);
REG_KERNEL_BASE(FftC2RC64Kernel);
}  // namespace AsdSip