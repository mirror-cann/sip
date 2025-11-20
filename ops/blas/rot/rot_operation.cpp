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
#include "rot.h"

using namespace Mki;

constexpr int64_t INPUT_NUM = 2;
constexpr int64_t OUTPUT_NUM = 0;

namespace AsdSip {
class RotOperation : public OperationBase {
public:
    explicit RotOperation(const std::string &opName) noexcept : OperationBase(opName) {}
    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Rot), "OpParam is invalid", return nullptr);
        OpParam::Rot param = AnyCast<OpParam::Rot>(launchParam.GetParam());
        switch (param.rotType) {
            case OpParam::Rot::RotType::ROT_CSROT:
                return GetKernelByName("CsrotC64Kernel");
            default:
                ASDSIP_LOG(ERROR) << "No kernel for Rot";
                return nullptr;
        }
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::Rot), "OpParam is invalid", return 0);
        return INPUT_NUM;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::Rot), "OpParam is invalid", return 0);
        return OUTPUT_NUM;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Rot), "OpParam is invalid",
                  return Status::FailStatus(ERROR_INVALID_VALUE));

        return Status::OkStatus();
    }
};
REG_OPERATION(RotOperation);
}  //    namespace AsdSip