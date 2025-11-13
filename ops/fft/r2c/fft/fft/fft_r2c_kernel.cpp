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
#include "../../../common/include/tiling/fft_all_mix_tiling_data.h"
#include "tiling/fft_r2c_tiling.h"

namespace AsdSip {
constexpr uint64_t TENSOR_PERM_IDX = 1;

class FftR2CKernel : public KernelBase {
public:
    explicit FftR2CKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FftAllMix), "FftR2C: param type invalid",
                  return false);
        ASDSIP_CHECK(launchParam.GetInTensorCount() == 8, "InTensor num invalid", return false);
        ASDSIP_CHECK(launchParam.GetOutTensorCount() == 1, "OutTensor num invalid", return false);  // markhere
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        uint64_t r2cTilingSize = static_cast<uint64_t>(sizeof(FftAllMixTilingData));
        return r2cTilingSize;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = FftR2CTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl FftR2CTiling failed",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        kernelInfo_.SetHwsyncIdx(0);
        return Status::OkStatus();
    }
};

// FftR2C1C64Kernel
class FftR2C1C64Kernel : public FftR2CKernel {
public:
    explicit FftR2C1C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftR2CKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftR2CKernel::CanSupport(launchParam), "FftR2C1C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

// FftR2C2C64Kernel
class FftR2C2C64Kernel : public FftR2CKernel {
public:
    explicit FftR2C2C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftR2CKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftR2CKernel::CanSupport(launchParam), "FftR2C2C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

// FftR2C3C64Kernel
class FftR2C3C64Kernel : public FftR2CKernel {
public:
    explicit FftR2C3C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftR2CKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftR2CKernel::CanSupport(launchParam), "FftR2C3C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

// FftR2COdd1C64Kernel
class FftR2COdd1C64Kernel : public FftR2CKernel {
public:
    explicit FftR2COdd1C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftR2CKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftR2CKernel::CanSupport(launchParam), "FftR2COdd1C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

// FftR2COdd2C64Kernel
class FftR2COdd2C64Kernel : public FftR2CKernel {
public:
    explicit FftR2COdd2C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftR2CKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftR2CKernel::CanSupport(launchParam), "FftR2COdd2C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

// FftR2COdd3C64Kernel
class FftR2COdd3C64Kernel : public FftR2CKernel {
public:
    explicit FftR2COdd3C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftR2CKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftR2CKernel::CanSupport(launchParam), "FftR2COdd3C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

// FftR2CEven1C64Kernel
class FftR2CEven1C64Kernel : public FftR2CKernel {
public:
    explicit FftR2CEven1C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftR2CKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftR2CKernel::CanSupport(launchParam), "FftR2CEven1C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

// FftR2CEven2C64Kernel
class FftR2CEven2C64Kernel : public FftR2CKernel {
public:
    explicit FftR2CEven2C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftR2CKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftR2CKernel::CanSupport(launchParam), "FftR2CEven2C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

// FftR2CEven3C64Kernel
class FftR2CEven3C64Kernel : public FftR2CKernel {
public:
    explicit FftR2CEven3C64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FftR2CKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(FftR2CKernel::CanSupport(launchParam), "FftR2CEven3C64 failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype unsupported",
                  return false);
        return true;
    }
};

REG_KERNEL_BASE(FftR2C1C64Kernel);
REG_KERNEL_BASE(FftR2C2C64Kernel);
REG_KERNEL_BASE(FftR2C3C64Kernel);
REG_KERNEL_BASE(FftR2COdd1C64Kernel);
REG_KERNEL_BASE(FftR2COdd2C64Kernel);
REG_KERNEL_BASE(FftR2COdd3C64Kernel);
REG_KERNEL_BASE(FftR2CEven1C64Kernel);
REG_KERNEL_BASE(FftR2CEven2C64Kernel);
REG_KERNEL_BASE(FftR2CEven3C64Kernel);
}  // namespace AsdSip