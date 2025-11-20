/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "utils/assert.h"
#include "log/log.h"
#include "ssyr.h"

using namespace Mki;

namespace AsdSip {
static constexpr int64_t TENSOR_INPUT_NUM = 1;
static constexpr int64_t TENSOR_OUTPUT_NUM = 1;

class SsyrOperation : public OperationBase {
public:
    explicit SsyrOperation(const std::string &opName) noexcept : OperationBase(opName) {}
    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        return GetKernelByName("SsyrF32Kernel");
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::Ssyr), "OpParam is invalid", return 0);
        return TENSOR_INPUT_NUM;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::Ssyr), "OpParam is invalid", return 0);
        return TENSOR_OUTPUT_NUM;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Ssyr), "OpParam is invalid",
                  return Status::FailStatus(ERROR_INVALID_VALUE));

        outTensors[0].desc.dtype = launchParam.GetOutTensor(0).desc.dtype;
        outTensors[0].desc.format = launchParam.GetOutTensor(0).desc.format;
        outTensors[0].desc.dims = launchParam.GetOutTensor(0).desc.dims;

        return Status::OkStatus();
    }
};
REG_OPERATION(SsyrOperation);
}  // namespace AsdSip