/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FFTCORE_COMMONFUNC_UTILS__
#define __FFTCORE_COMMONFUNC_UTILS__

#include <cstdint>
#include <cstddef>

const int L0AB_PINGPONG_BUFFER_LEN = 128 * 64;

int64_t ROUND(int64_t num, int64_t pad);

int64_t MIN(int64_t num1, int64_t num2);

void getTile(int64_t N1, int64_t N2, int64_t stepIndex, int64_t stepLen,
             int32_t &tileM0, int32_t &tileN0, int32_t &tileK0);

bool checkSizeToMalloc(size_t size);

#endif  // __FFTCORE_COMMONFUNC_UTILS__
