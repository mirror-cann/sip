/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mki/base/operation_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/SVector/SVector.h>
#include "utils/assert.h"
#include "log/log.h"
#include "fft_all_mix.h"

namespace {
constexpr int EVEN_THRESHOLD_N = 524288 / 2;
}

namespace AsdSip {
using namespace Mki;
constexpr int64_t INPUT_NUM = 8;
constexpr int AIVWAYC2R2C = 2;
constexpr int AIVWAYC2R3C = 3;
class FftC2ROperation : public OperationBase {
public:
    explicit FftC2ROperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        const auto &param = AnyCast<OpParam::FftAllMix>(launchParam.GetParam());

        if (param.aiv_split_way == 1 && param.parity == 1) {
            return GetKernelByName("FftC2ROdd1C64Kernel");
        } else if (param.aiv_split_way == 1 && param.parity == 0) {
            return param.fftN <= EVEN_THRESHOLD_N ? GetKernelByName("FftC2REven1C64Kernel") :
                                                  GetKernelByName("FftC2R1C64Kernel");
        } else if (param.aiv_split_way == AIVWAYC2R2C && param.parity == 1) {
            return GetKernelByName("FftC2ROdd2C64Kernel");
        } else if (param.aiv_split_way == AIVWAYC2R2C && param.parity == 0) {
            return param.fftN <= EVEN_THRESHOLD_N ? GetKernelByName("FftC2REven2C64Kernel") :
                                                  GetKernelByName("FftC2R2C64Kernel");
        } else if (param.aiv_split_way == AIVWAYC2R3C && param.parity == 1) {
            return GetKernelByName("FftC2ROdd3C64Kernel");
        } else if (param.aiv_split_way == AIVWAYC2R3C && param.parity == 0) {
            return param.fftN <= EVEN_THRESHOLD_N ? GetKernelByName("FftC2REven3C64Kernel") :
                                                  GetKernelByName("FftC2R3C64Kernel");
        }

        return nullptr;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FftAllMix), "OpParam is invalid",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "no match specificParam type."));
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "error dataDype",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        const auto &param = AnyCast<OpParam::FftAllMix>(launchParam.GetParam());
        outTensors[0].desc.dtype = TENSOR_DTYPE_FLOAT;
        outTensors[0].desc.format = launchParam.GetInTensor(0).desc.format;
        int64_t batch = launchParam.GetInTensor(0).desc.dims[0];
        outTensors[0].desc.dims = {batch, param.N_doing};

        return Status::OkStatus();
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::FftAllMix), "OpParam is invalid", return 0);
        return INPUT_NUM;
    }
};
REG_OPERATION(FftC2ROperation);
}  //    namespace AsdSip