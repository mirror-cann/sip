/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mki/base/operation_base.h"
#include "utils/assert.h"
#include "log/log.h"
#include "mki_loader/op_register.h"
#include "fft_all_mix.h"
#include "mki/utils/SVector/SVector.h"

namespace AsdSip {
using namespace Mki;
constexpr int32_t AIV_SPLIT_WAY_MIX1C = 1;
constexpr int32_t AIV_SPLIT_WAY_MIX2C = 2;
constexpr int32_t AIV_SPLIT_WAY_MIX3C = 3;
constexpr int32_t AIV_SPLIT_WAY_MIX4C = 4;
constexpr int32_t AIV_SPLIT_WAY_MIX5C = 5;
constexpr int32_t AIV_SPLIT_WAY_MIX6C = 6;
constexpr int64_t INPUT_NUM = 4;
class FftMixOperation : public OperationBase {
public:
    explicit FftMixOperation(const std::string &opName) noexcept : OperationBase(opName) {}
    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        const auto &param = AnyCast<OpParam::FftAllMix>(launchParam.GetParam());

        if (param.aiv_split_way == AIV_SPLIT_WAY_MIX1C) {
            return GetKernelByName("FftMix1C64Kernel");
        } else if (param.aiv_split_way == AIV_SPLIT_WAY_MIX2C) {
            return GetKernelByName("FftMix2C64Kernel");
        } else if (param.aiv_split_way == AIV_SPLIT_WAY_MIX3C) {
            return GetKernelByName("FftMix3C64Kernel");
        } else if (param.aiv_split_way == AIV_SPLIT_WAY_MIX4C) {
            return GetKernelByName("FftMix4C64Kernel");
        } else if (param.aiv_split_way == AIV_SPLIT_WAY_MIX5C) {
            return GetKernelByName("FftMix5C64Kernel");
        } else if (param.aiv_split_way == AIV_SPLIT_WAY_MIX6C) {
            return GetKernelByName("FftMix6C64Kernel");
        }

        return nullptr;
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::FftAllMix), "OpParam is invalid", return 0);
        return INPUT_NUM;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FftAllMix), "OpParam is invalid",
                  return Status::FailStatus(ERROR_INVALID_VALUE, "no match specificParam type."));

        outTensors[0].desc.dtype = launchParam.GetInTensor(0).desc.dtype;
        outTensors[0].desc.format = launchParam.GetInTensor(0).desc.format;
        outTensors[0].desc.dims = launchParam.GetInTensor(0).desc.dims;

        return Status::OkStatus();
    }
};
REG_OPERATION(FftMixOperation);
}  //    namespace AsdSip