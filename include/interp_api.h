/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_INTERP_API_H
#define ASDSIP_INTERP_API_H

#include "acl/acl.h"
#include "utils/aspb_status.h"
#include "utils/mem_base.h"

namespace AsdSip {

AspbStatus asdInterpWithCoeff(
    const aclTensor *x, const aclTensor *coefficient, aclTensor *output, void *stream, void *workSpace = nullptr);
AspbStatus asdInterpWithCoeffGetWorkspaceSize(size_t &workspaceSize);

}  // namespace AsdSip
#endif