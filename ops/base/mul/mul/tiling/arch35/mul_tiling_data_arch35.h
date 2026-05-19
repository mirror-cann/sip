/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASDSIP_MUL_TILING_DATA_ARCH35
#define ASDSIP_MUL_TILING_DATA_ARCH35

#include <cstdint>

namespace AsdSip {

constexpr int64_t DTYPE_C64_ARCH35 = 0;
constexpr int64_t DTYPE_C32_ARCH35 = 1;

struct CMulArch35TilingData {
    int64_t n;
    int64_t dtype;
};

}  // namespace AsdSip

#endif  // ASDSIP_MUL_TILING_DATA_ARCH35