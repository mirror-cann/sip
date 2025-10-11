
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sscal_aiv.h"

#ifdef __DAV_C220_VEC__
extern "C" __global__ __aicore__ void sscal(GM_ADDR x, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    Sscal::SscalAIV<float> op;
    op.Init(x, workSpace, tilingGm);
    op.Process();
}
#endif