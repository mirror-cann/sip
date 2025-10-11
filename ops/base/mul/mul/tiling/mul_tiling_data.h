/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_MULTINOMIAL_TILING_DATA
#define ASDSIP_MULTINOMIAL_TILING_DATA

#include <cstdint>

namespace AsdSip {
struct CMulTilingData {
    int64_t n;
    int64_t useCoreNum;
    int64_t calNum[48];
    int64_t startOffset[48];
};
}  // namespace AsdSip
#endif