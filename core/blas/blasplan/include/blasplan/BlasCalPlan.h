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

#include "BlasPlan.h"

namespace AsdSip {
class BlasCalPlan : public BlasPlan {
public:
    BlasCalPlan();
    ~BlasCalPlan() override;
    AsdSip::AspbStatus CreateTensor() override;
    AsdSip::AspbStatus SetMaskTensor();
    AsdSip::AspbStatus FreeTensor() override;
    Mki::Tensor GetMaskTensor()
    {
        return this->maskTensor;
    }
    aclTensor *GetAclMaskTensor()
    {
        return this->maskAclTensor;
    }
    int64_t GetWorkspaceSize() override;

private:
    Mki::Tensor maskTensor;
    aclTensor *maskAclTensor = nullptr;
    int64_t m;
    int64_t n;
    int64_t incy;
    char transA;
};
}
