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
class BlasCtrmvPlan : public BlasPlan {
public:
    explicit BlasCtrmvPlan(asdBlasFillMode_t uplo, int64_t n);
    ~BlasCtrmvPlan() override;
    AsdSip::AspbStatus CreateTensor() override;
    AsdSip::AspbStatus SetUploTensor();
    AsdSip::AspbStatus FreeTensor() override;
    Mki::Tensor GetUploTensor()
    {
        return this->uploTensor;
    }
    aclTensor *GetUploAclTensor()
    {
        return this->uploAclTensor;
    }
    int64_t GetWorkspaceSize() override;

private:
    Mki::Tensor uploTensor;
    aclTensor *uploAclTensor = nullptr;
    asdBlasFillMode_t uplo;
    int64_t n;
};
}
