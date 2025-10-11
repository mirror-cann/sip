/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include "mki/base/operation_base.h"
#include "utils/assert.h"
#include "mki_loader/op_register.h"
#include "utils/op_desc.h"
#include "interpbycoeff.h"

namespace AsdSip {
using namespace Mki;

constexpr int64_t INPUT_NUM = 2;
constexpr int64_t OUTPUT_NUM = 1;

class InterpByCoeffOperation : public OperationBase {
public:
    explicit InterpByCoeffOperation(const std::string &opName) noexcept : OperationBase(opName) {}
    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        return GetKernelByName("InterpByCoeffKernel");
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::InterpByCoeff), "OpParam is invalid", return 0);
        return INPUT_NUM;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::InterpByCoeff), "OpParam is invalid", return 0);
        return OUTPUT_NUM;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::InterpByCoeff), "OpParam is invalid",
                  return Status::FailStatus(ERROR_INVALID_VALUE));

        OpParam::InterpByCoeff param = AnyCast<OpParam::InterpByCoeff>(launchParam.GetParam());

        outTensors[0].desc = launchParam.GetInTensor(0).desc;
        outTensors[0].desc.dims[1] = param.interpLength;

        return Status::OkStatus();
    }
};
REG_OPERATION(InterpByCoeffOperation);
}  //    namespace AsdSip