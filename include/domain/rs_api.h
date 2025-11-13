/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_SIGNAL_API_H
#define ASDSIP_SIGNAL_API_H

#include "utils/aspb_status.h"
#include "acl/acl.h"
#include "utils/mem_base.h"

namespace AsdSip {

AspbStatus rsInterpolationBySinc(const aclTensor *inputTensor, const aclTensor *sincTab,
                                 const aclTensor *posFloor, const aclTensor *posToTabIndex,
                                 aclTensor *outputTensor, int interpNum, int quantNum, int interpLength,
                                 void *stream, void *workSpace = nullptr);

AspbStatus rsInterpolationBySincGetWorkspaceSize(size_t &workspaceSize);

}
#endif