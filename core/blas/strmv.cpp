/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "strmv.h"
#include "blas_common.h"
#include "blasplan/include/blasplan/BlasStrmvPlan.h"

using namespace AsdSip;
using namespace Mki;

namespace AsdSip {
struct StrmvTensorParam {
    const aclTensor *A;
    const aclTensor *x;
};

AspbStatus StrmvDtypeCheck(struct StrmvTensorParam parm)
{
    auto ret = ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE;
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.A, aclDataType::ACL_FLOAT, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.x, aclDataType::ACL_FLOAT, ret);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus StrmvShapeCheck(struct StrmvTensorParam parm, const int64_t n)
{
    ASDSIP_ECHECK(
        n > 0 && n <= UINT32_MAX, "blas asdBlasStrmv get n <= 0 or n > 2^32.", ErrorType::ACL_ERROR_INVALID_PARAM);
    SIP_OP_CHECK_NUM_NOT_MATCH(parm.A, n * n, ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    SIP_OP_CHECK_NUM_NOT_MATCH(parm.x, n, ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdBlasStrmv(asdBlasHandle handle, asdBlasFillMode_t uplo, asdBlasOperation_t trans, asdBlasDiagType_t diag,
    const int64_t n, aclTensor *A, const int64_t lda, aclTensor *x, const int64_t incx)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle),
        "blas Strmv get cached plan failed.",
        ErrorType::ACL_ERROR_INTERNAL_ERROR);

    struct StrmvTensorParam parm {
        A, x
    };
    ASDSIP_CHECK(StrmvDtypeCheck(parm) == ErrorType::ACL_SUCCESS,
        "blas asdBlasStrmv dtype check failed.",
        return ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    auto ret = StrmvShapeCheck(parm, n);
    ASDSIP_CHECK(ret == ErrorType::ACL_SUCCESS, "blas asdBlasStrmv shape check failed.", return ret);

    if (incx <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasStrmv get incx <= 0.";
    }

    try {
        AsdSip::BlasStrmvPlan &plan = dynamic_cast<AsdSip::BlasStrmvPlan &>(BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "Strmv plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        OpDesc opDesc;
        opDesc.opName = "StrmvOperation";
        AsdSip::OpParam::Strmv param = {0, 0, 0, 0, 0};

        if (trans == asdBlasOperation_t::ASDBLAS_OP_N) {
            param.trans = 0;
        } else {
            param.trans = 1;
        }

        if (uplo == asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER) {
            param.uplo = 1;
        } else {
            param.uplo = 0;
        }

        if (diag == asdBlasDiagType_t::ASDBLAS_DIAG_UNIT) {
            param.diag = 1;
        } else {
            param.diag = 0;
        }
        param.incx = 1;
        param.lda = n;
        opDesc.specificParam = param;
        ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

        SVector<aclTensor *> inTensors{A, x, plan.GetmMaskAclTensor()};
        SVector<aclTensor *> outTensors{x};

        Status status = RunAsdOpsV2(plan.GetStream(), opDesc, inTensors, outTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasStrmv success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast &e) {
        // Handle the op strmv exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Strmv Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasMakeStrmvPlan(asdBlasHandle handle, asdBlasFillMode_t uplo, asdBlasOperation_t trans, int64_t n)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas MakeStrmvPlan Fail.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

    AsdSip::BlasStrmvPlan *plan = nullptr;
    try {
        plan = new AsdSip::BlasStrmvPlan(uplo, trans, n);
        BlasPlanCache::MakePlan(handle, plan);
    } catch (const std::exception &e) {
        if (plan != nullptr) {
            delete plan;
        }
        delete static_cast<int *>(handle);
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make Strmv Plan failed: " << e.what();
        throw std::runtime_error("Make Strmv Plan failed.");
    }
    if (plan->CreateTensor() != ErrorType::ACL_SUCCESS) {
        plan->MarkFailed();
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Fail to create blas asdBlasMakeStrmvPlan mask tensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
}  // namespace AsdSip
