/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <mki/utils/SVector/SVector.h>
#include "BlasPlan.h"
#include "acl/acl.h"

namespace AsdSip {
struct CgemmPlanParam {
    asdBlasOperation_t transa;
    asdBlasOperation_t transb;
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
};

class BlasCgemmPlan : public BlasPlan {
public:
    Mki::SVector<Mki::Tensor> augATensors = {};
    Mki::SVector<Mki::Tensor> augBTensors = {};
    Mki::SVector<Mki::Tensor> augCTensors = {};
 
    Mki::SVector<aclTensor *> augAAclTensors = {};
    Mki::SVector<aclTensor *> augBAclTensors = {};
    Mki::SVector<aclTensor *> augCAclTensors = {};

public:
    explicit BlasCgemmPlan(CgemmPlanParam planParam);
    AsdSip::AspbStatus CreateTensor() override;
    void FreeGivenTensor(Mki::Tensor tensor) const;
    void FreeAclTensors(Mki::SVector<aclTensor *> aclTensorList) const;
    AsdSip::AspbStatus FreeTensor() override;
    int64_t GetWorkspaceSize() override;
    BlasCgemmPlan();
    ~BlasCgemmPlan() override;

private:
    CgemmPlanParam planParam;
};
}
