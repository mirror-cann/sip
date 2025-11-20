/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include "mki/base/operation_base.h"
#include "utils/assert.h"
#include "log/log.h"
#include "mki_loader/op_register.h"
#include "mki/utils/SVector/SVector.h"
#include "utils/op_desc.h"
#include "cgemm_batched.h"


namespace AsdSip {
using namespace Mki;

constexpr int64_t INPUT_NUM = 3;
constexpr int64_t OUTPUT_NUM = 1;
constexpr int64_t TWO = 2;

class CgemmBatchedOperation : public OperationBase {
public:
    explicit CgemmBatchedOperation(const std::string &opName) noexcept : OperationBase(opName) {}
    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        return GetKernelByName("CgemmBatchedC64Kernel");
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::CgemmBatched), "OpParam is invalid", return 0);
        return INPUT_NUM;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::CgemmBatched), "OpParam is invalid", return 0);
        return OUTPUT_NUM;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::CgemmBatched), "OpParam is invalid",
                  return Status::FailStatus(ERROR_INVALID_VALUE));

        OpParam::CgemmBatched param = AnyCast<OpParam::CgemmBatched>(launchParam.GetParam());

        outTensors[0].desc = launchParam.GetInTensor(0).desc;
        outTensors[0].desc.dims[0] = param.batch;
        outTensors[0].desc.dims[1] = param.m;
        outTensors[0].desc.dims[TWO] = param.n;

        return Status::OkStatus();
    }
};
REG_OPERATION(CgemmBatchedOperation);
}  //    namespace AsdSip