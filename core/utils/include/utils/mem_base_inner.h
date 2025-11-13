/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_MEM_BASE_INNER_H
#define ASDSIP_MEM_BASE_INNER_H

#include "mki/utils/rt/rt.h"
#include "mki/tensor.h"
#include "mki/utils/status/status.h"

typedef struct aclTensor aclTensor;

Mki::Status MallocTensorInDevice(Mki::Tensor &tensor);

Mki::Status CopyOutTensorToHost(Mki::Tensor &tensor);

Mki::Status FreeTensorInDevice(const Mki::Tensor &tensor);

Mki::Status setGlobalWorkspace(uint8_t *buffer, uint64_t bufferSize);

Mki::Status toAclTensor(const Mki::Tensor &inTensor, aclTensor *&outTensor, std::vector<int64_t> stride = {});

#endif