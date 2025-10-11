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
#include <utils/op_desc.h>
#include "utils/assert.h"
#include "log/log.h"
#include "cgemm.h"

using namespace Mki;

namespace AsdSip {

constexpr int64_t INPUT_DIM = 10;
constexpr int64_t OUTPUT_DIM = 1;

class CgemmOperation : public OperationBase {
public:
    explicit CgemmOperation(const std::string &opName) noexcept : OperationBase(opName) {}
    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        return GetKernelByName("CgemmC64Kernel");
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::Cgemm), "OpParam is invalid", return 0);
        return INPUT_DIM;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::Cgemm), "OpParam is invalid", return 0);
        return OUTPUT_DIM;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Cgemm), "Cgemm OpParam is invalid",
                  return Status::FailStatus(ERROR_INVALID_VALUE));

        outTensors[0].desc.dtype = TENSOR_DTYPE_COMPLEX64;
        outTensors[0].desc.format = launchParam.GetOutTensor(0).desc.format;
        outTensors[0].desc.dims = launchParam.GetOutTensor(0).desc.dims;

        return Status::OkStatus();
    }
};
REG_OPERATION(CgemmOperation);
}  //    namespace AsdSip