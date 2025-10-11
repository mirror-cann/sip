/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "utils/assert.h"
#include "log/log.h"
#include "base_api.h"
#include "utils/ops_base.h"
#include "mul.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn/opdev/common_types.h"

using namespace Mki;
using namespace AsdSip;

namespace AsdSip {
AspbStatus asdMul(int n, const aclTensor *x, const aclTensor *y, aclTensor *z, void *stream, void *workspace)
{
    int64_t *storageDims = nullptr;
    uint64_t storageDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(x, &storageDims, &storageDimsNum), "asdMul: aclGetStorageShape");
    int64_t sizeX = (*storageDims) * static_cast<int64_t>(storageDimsNum);

    delete[] storageDims;
    storageDims = nullptr;

    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(y, &storageDims, &storageDimsNum), "asdMul: aclGetStorageShape");
    int64_t sizeY = (*storageDims) * static_cast<int64_t>(storageDimsNum);

    delete[] storageDims;
    storageDims = nullptr;

    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(z, &storageDims, &storageDimsNum), "asdMul: aclGetStorageShape");
    int64_t sizeZ = (*storageDims) * static_cast<int64_t>(storageDimsNum);

    if (storageDims != nullptr) {
        delete[] storageDims;
        storageDims = nullptr;
    }

    op::DataType dataType = op::DataType::DT_UNDEFINED;
    dataType = x->GetDataType();
    ASDSIP_ECHECK(dataType == op::DataType::DT_COMPLEX64 || dataType == op::DataType::DT_COMPLEX32,
        "base asdMul get wrong x tensor dtype.",
        ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    dataType = op::DataType::DT_UNDEFINED;
    dataType = y->GetDataType();
    ASDSIP_ECHECK(dataType == op::DataType::DT_COMPLEX64 || dataType == op::DataType::DT_COMPLEX32,
        "base asdMul get wrong y tensor dtype.",
        ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    dataType = op::DataType::DT_UNDEFINED;
    dataType = z->GetDataType();
    ASDSIP_ECHECK(dataType == op::DataType::DT_COMPLEX64 || dataType == op::DataType::DT_COMPLEX32,
        "base asdMul get wrong z tensor dtype.",
        ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    ASDSIP_ECHECK(n > 0, "base asdMul get n <= 0.", ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(n == sizeX && n == sizeY && n == sizeZ,
        "Invalid tensor dimensions: parameter 'n' must be equal to sizeX, sizeY, and sizeZ.",
        ErrorType::ACL_ERROR_INVALID_PARAM);

    OpDesc opDesc;
    opDesc.opName = "MulOperation";
    AsdSip::OpParam::CMul param;
    param.n = n;
    param.cMulType = (dataType == op::DataType::DT_COMPLEX64) ? OpParam::CMul::CMulType::MUL_C64
                     : OpParam::CMul::CMulType::MUL_C32;
    opDesc.specificParam = param;
    ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

    SVector<aclTensor *> inTensors{const_cast<aclTensor *>(x), const_cast<aclTensor *>(y)};
    SVector<aclTensor *> outTensors{z};

    Status status = RunAsdOpsV2(stream, opDesc, inTensors, outTensors, (uint8_t *)workspace);
    ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

    ASDSIP_LOG(INFO) << "Execute asdBaseMul success.";
    return ErrorType::ACL_SUCCESS;
}
}  // namespace AsdSip