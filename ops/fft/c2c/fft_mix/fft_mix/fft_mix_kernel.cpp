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
#include "fft_all_mix.h"
#include "tiling/fft_mix_tiling.h"
#include "../../../common/include/tiling/fft_all_mix_tiling_data.h"

static constexpr uint32_t TENSOR_INPUT_NUM = 4;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 1;
namespace AsdSip {
using namespace Mki;
class FftMixKernel : public KernelBase {
public:
    explicit FftMixKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FftAllMix), "FftMix: param type invalid",
                  return false);
        ASDSIP_CHECK(launchParam.GetInTensorCount() == 4, "input num invalid", return false);
        ASDSIP_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);  // markhere
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(FftAllMixTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = FftMixTiling(launchParam, kernelInfo_);
        kernelInfo_.SetHwsyncIdx(0);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitKernelInfoImpl FftMixTiling failed",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
        return Status::OkStatus();
    }
};

// FftMix1C64Kernel
class FftMix1C64Kernel : public FftMixKernel {
public:
    explicit FftMix1C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftMixKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftMixKernel::CanSupport(launchParam), "FftMix1C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

// FftMix2C64Kernel
class FftMix2C64Kernel : public FftMixKernel {
public:
    explicit FftMix2C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftMixKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftMixKernel::CanSupport(launchParam), "FftMix2C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

// FftMix3C64Kernel
class FftMix3C64Kernel : public FftMixKernel {
public:
    explicit FftMix3C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftMixKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftMixKernel::CanSupport(launchParam), "FftMix3C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

// FftMix4C64Kernel
class FftMix4C64Kernel : public FftMixKernel {
public:
    explicit FftMix4C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftMixKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftMixKernel::CanSupport(launchParam), "FftMix4C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

// FftMix5C64Kernel
class FftMix5C64Kernel : public FftMixKernel {
public:
    explicit FftMix5C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftMixKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftMixKernel::CanSupport(launchParam), "FftMix5C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

// FftMix6C64Kernel
class FftMix6C64Kernel : public FftMixKernel {
public:
    explicit FftMix6C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftMixKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftMixKernel::CanSupport(launchParam), "FftMix6C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

REG_KERNEL_BASE(FftMix1C64Kernel);
REG_KERNEL_BASE(FftMix2C64Kernel);
REG_KERNEL_BASE(FftMix3C64Kernel);
REG_KERNEL_BASE(FftMix4C64Kernel);
REG_KERNEL_BASE(FftMix5C64Kernel);
REG_KERNEL_BASE(FftMix6C64Kernel);

}  // namespace AsdSip