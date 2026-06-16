/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mki/base/operation_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/SVector/SVector.h>
#include <mki/utils/platform/platform_info.h>
#include "utils/assert.h"
#include "log/log.h"
#include "fft_all_mix.h"
#include "fft_c2c_arch35.h"

namespace AsdSip {
using namespace Mki;

class FftC2CArch35Operation : public OperationBase {
public:
    explicit FftC2CArch35Operation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
            const auto &param = AnyCast<OpParam::FftC2CArch35>(launchParam.GetParam());
            return GetKernelByName(param.isMixedRadix ? "FftC2CArch35MixC64Kernel" : "FftC2CArch35C64Kernel");
        }
        return nullptr;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
            ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FftC2CArch35), "OpParam is invalid",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "no match specificParam type."));

            outTensors[0].desc.dtype = TENSOR_DTYPE_COMPLEX64;
            outTensors[0].desc.format = launchParam.GetInTensor(0).desc.format;
            int64_t batch = launchParam.GetInTensor(0).desc.dims[0];
            outTensors[0].desc.dims = {batch, *(launchParam.GetInTensor(0).desc.dims.end() - 1)};
        }

        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "error dataDype",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        return Status::OkStatus();
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
            ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::FftC2CArch35), "OpParam is invalid", return 0);
            return 4;
        }
        return 1;
    }
};

REG_OPERATION(FftC2CArch35Operation);
}  // namespace AsdSip
