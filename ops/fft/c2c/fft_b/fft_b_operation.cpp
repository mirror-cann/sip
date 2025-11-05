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
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/SVector/SVector.h>
#include "utils/assert.h"
#include "log/log.h"
#include "fft_b.h"

namespace AsdSip {

using Mki::Tensor;
using Mki::SVector;
using Mki::Status;
using Mki::Any;
using Mki::Kernel;
using Mki::LaunchParam;
using Mki::OperationBase;

constexpr int64_t INPUT_NUM = 4;

class FftBOperation : public OperationBase {
public:
    explicit FftBOperation(const std::string &opName) noexcept : OperationBase(opName) {}
    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);

        if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_910B) {
            return GetKernelByName("FftBC64Kernel");
        }

        return nullptr;
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::FftB), "OpParam is invalid", return 0);
        return INPUT_NUM;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FftB), "OpParam is invalid",
                  return Status::FailStatus(ERROR_INVALID_VALUE));

        outTensors[0].desc.dtype = launchParam.GetInTensor(0).desc.dtype;
        outTensors[0].desc.format = launchParam.GetInTensor(0).desc.format;
        outTensors[0].desc.dims = launchParam.GetInTensor(0).desc.dims;

        return Status::OkStatus();
    }
};
REG_OPERATION(FftBOperation);
}  //    namespace AsdSip