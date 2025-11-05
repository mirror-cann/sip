/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "utils/assert.h"
#include "mki/base/operation_base.h"
#include "log/log.h"
#include "mki_loader/op_register.h"
#include "mki/utils/SVector/SVector.h"
#include "conj.h"

namespace AsdSip {
using namespace Mki;
class ConjOperation : public OperationBase {
public:
    explicit ConjOperation(const std::string &opName) noexcept : OperationBase(opName) {}
    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        return GetKernelByName("ConjC64Kernel");
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        const Any &specificParam = launchParam.GetParam();
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::Conj), "OpParam is invalid",
                  return Status::FailStatus(ERROR_INVALID_VALUE));

        outTensors[0].desc.dtype = launchParam.GetInTensor(0).desc.dtype;
        outTensors[0].desc.format = launchParam.GetInTensor(0).desc.format;
        outTensors[0].desc.dims = launchParam.GetInTensor(0).desc.dims;

        return Status::OkStatus();
    }
};
REG_OPERATION(ConjOperation);
}  //    namespace AsdSip