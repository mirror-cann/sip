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
class BlasHCgemmBatchedPlan : public BlasPlan {
public:
    AsdSip::AspbStatus CreateTensor() override;
    AsdSip::AspbStatus FreeTensor() override;
    int64_t GetWorkspaceSize() override;
    BlasHCgemmBatchedPlan();
    ~BlasHCgemmBatchedPlan() override;

public:
    aclTensor* gatherOffsets{nullptr};
};

// for complex<float>
class BlasCgemmBatchedPlan : public BlasPlan {
public:
    AsdSip::AspbStatus CreateTensor() override;
    AsdSip::AspbStatus FreeTensor() override;
    int64_t GetWorkspaceSize() override;
    BlasCgemmBatchedPlan();
    ~BlasCgemmBatchedPlan() override;

public:
    aclTensor* gatherOffsets{nullptr};
};

}
