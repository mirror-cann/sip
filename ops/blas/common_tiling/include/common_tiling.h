/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_OPS_COMMON_TILING_H
#define ASDSIP_OPS_COMMON_TILING_H

#include <cstdint>
#include "common_tiling_data.h"
namespace AsdSip {
uint32_t ConfigCommonTilingData(CommonTilingData *tilingDataPtr, uint32_t totalEleNum, uint32_t vecCoreNum);
}  // namespace AsdSip

#endif  // ASCEND_OPS_COMMON_TILING_H