/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "mki/tensor.h"
#include "blas_api.h"
#include "utils/aspb_status.h"

namespace AsdSip {
enum class AsdBlasPlanStatus {
    PLAN_UNINITIALIZED = 0,
    PLAN_INITIALIZED = 1,
    PLAN_FAILED = 2
};

class BlasPlan {
public:
    BlasPlan();
    virtual ~BlasPlan();
    void MarkInitialized();
    void MarkFailed();
    bool IsInitialized() const;

    virtual int64_t GetWorkspaceSize();
    void SetWorkspace(uint8_t *workspacePtr);
    uint8_t *GetWorkspace();
    bool HasWorkSpace() const;

    virtual AsdSip::AspbStatus CreateTensor();
    virtual AsdSip::AspbStatus FreeTensor();

    void *GetStream();
    void SetStream(void *asdBalsStream);

    void DestroyPlanData();

protected:
    AsdBlasPlanStatus blasStatus;
    void *stream;
    uint8_t *workspace = nullptr;
};

}
