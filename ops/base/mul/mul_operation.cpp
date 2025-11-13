/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <utils/op_desc.h>
#include "utils/assert.h"
#include "log/log.h"
#include "mki/base/operation_base.h"
#include "mki_loader/op_register.h"
#include "mki/utils/SVector/SVector.h"
#include "mul.h"

namespace AsdSip {
using namespace Mki;
static constexpr uint32_t TENSOR_INPUT_NUM = 2;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 1;
class MulOperation : public OperationBase {
public:
    explicit MulOperation(const std::string &opName) noexcept : OperationBase(opName)
    {}
    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        OpParam::CMul param = AnyCast<OpParam::CMul>(launchParam.GetParam());
        if (param.cMulType == OpParam::CMul::CMulType::MUL_C64) {
            return GetKernelByName("MulC64Kernel");
        } else if (param.cMulType == OpParam::CMul::CMulType::MUL_C32) {
            return GetKernelByName("MulC32Kernel");
        } else {
            ASDSIP_LOG(ERROR) << "cMulType is "<< static_cast<int>(param.cMulType) << " .No kernel for asdMul";
            return nullptr;
        }
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::CMul), "OpParam is invalid", return 0);
        return TENSOR_INPUT_NUM;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::CMul), "OpParam is invalid", return 0);
        return TENSOR_OUTPUT_NUM;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        const Any &specificParam = launchParam.GetParam();
        MKI_CHECK(specificParam.Type() == typeid(OpParam::CMul),
            "OpParam is invalid",
            return Status::FailStatus(ERROR_INVALID_VALUE));

        OpParam::CMul param = AnyCast<OpParam::CMul>(launchParam.GetParam());
        if (param.cMulType == OpParam::CMul::CMulType::MUL_C64) {
            outTensors[0].desc.dtype = TENSOR_DTYPE_COMPLEX64;
            outTensors[0].desc.format = launchParam.GetInTensor(0).desc.format;
            outTensors[0].desc.dims = launchParam.GetInTensor(0).desc.dims;
        } else {
            outTensors[0].desc.dtype = TENSOR_DTYPE_COMPLEX32;
            outTensors[0].desc.format = launchParam.GetInTensor(0).desc.format;
            outTensors[0].desc.dims = launchParam.GetInTensor(0).desc.dims;
        }

        return Status::OkStatus();
    }
};
REG_OPERATION(MulOperation);
}  // namespace AsdSip