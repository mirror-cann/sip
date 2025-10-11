/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
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
#include "cal.h"

static constexpr int64_t CSCAL_TENSOR_INPUT_NUM = 2;
static constexpr int64_t SSCAL_TENSOR_INPUT_NUM = 1;
static constexpr int64_t TENSOR_OUTPUT_NUM = 0;

using namespace Mki;

namespace AsdSip {
class CalOperation : public OperationBase {
public:
    explicit CalOperation(const std::string &opName) noexcept : OperationBase(opName) {}
    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Cal), "OpParam is invalid", return nullptr);
        OpParam::Cal param = AnyCast<OpParam::Cal>(launchParam.GetParam());
        switch (param.calType) {
            case OpParam::Cal::CalType::CAL_SSCAL:
            case OpParam::Cal::CalType::CAL_CSSCAL:
                return GetKernelByName("SscalF32Kernel");
            case OpParam::Cal::CalType::CAL_CSCAL:
                return GetKernelByName("CscalC64Kernel");
            default:
                ASDSIP_LOG(ERROR) << "No kernel for Cal";
                return nullptr;
        }
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::Cal), "OpParam is invalid", return 0);
        OpParam::Cal param = AnyCast<OpParam::Cal>(specificParam);
        if (param.calType == OpParam::Cal::CalType::CAL_CSCAL) {
            return CSCAL_TENSOR_INPUT_NUM;
        }
        return SSCAL_TENSOR_INPUT_NUM;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::Cal), "OpParam is invalid", return 0);
        return TENSOR_OUTPUT_NUM;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Cal), "Cal OpParam is invalid",
                  return Status::FailStatus(ERROR_INVALID_VALUE));

        return Status::OkStatus();
    }
};
REG_OPERATION(CalOperation);
}  //    namespace AsdSip