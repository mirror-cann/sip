/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_FILTER_API_H
#define ASDSIP_FILTER_API_H

#include "utils/aspb_status.h"
#include "acl/acl.h"
#include "utils/mem_base.h"


namespace AsdSip {

enum class asdConvolveMode_t {
    ASD_CONVOLVE_FULL = 0,
    ASD_CONVOLVE_SAME = 1,
    ASD_CONVOLVE_VALID = 2
};


AspbStatus asdConvolveGetWorkspaceSize(int64_t signalLen, int64_t kernelLen, size_t &size);

AspbStatus asdConvolve(const aclTensor *signal, const aclTensor *kernel, aclTensor *output,
    asdConvolveMode_t mode, void *stream, void *workspace = nullptr);

}
#endif
