/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "fftcore/fft_core_common_func_utils.h"

constexpr int32_t N1_MAX45 = 45;
constexpr int32_t N1_MAX64 = 64;
constexpr int32_t N2_MAX8 = 8;
constexpr int32_t ROUND_16 = 16;
constexpr int32_t TITLE_CONST = 128;
constexpr int32_t CACL_TWO = 2;
constexpr int32_t STEP_LEN_THREE = 3;

int64_t ROUND(int64_t num, int64_t pad)
{
    if (pad == 0) {
        return 1;
    }
    return (num + pad - 1) / pad * pad;
}

int64_t MIN(int64_t num1, int64_t num2)
{
    return (num1 < num2) ? num1 : num2;
}

void getTile(
    int64_t N1,
    int64_t N2,
    int64_t stepIndex,
    int64_t stepLen,
    int32_t &tileM0,
    int32_t &tileN0,
    int32_t &tileK0
)
{
    if (N1 <= N1_MAX45 || (N1 <= N1_MAX64 && N2 <= N2_MAX8)) {
        tileM0 = ROUND(CACL_TWO * N1, ROUND_16);
        tileK0 = tileM0;
        if (stepIndex == stepLen - 1) {
            tileK0 = CACL_TWO * ROUND(N1, ROUND_16);
        }

        if (tileK0 == 0) {
            throw std::runtime_error("tileK0 is 0!");
        }
        tileN0 = L0AB_PINGPONG_BUFFER_LEN * CACL_TWO / tileK0 / ROUND_16 * ROUND_16;
        tileN0 = MIN(tileN0, ROUND(N2, ROUND_16));
    } else {
        tileM0 = TITLE_CONST;
        tileN0 = TITLE_CONST;
        tileK0 = TITLE_CONST;
        if (stepIndex == stepLen - 1) {
            if (tileK0 > CACL_TWO * ROUND(N1, ROUND_16) / CACL_TWO && tileK0 < CACL_TWO * ROUND(N1, ROUND_16)) {
                tileK0 = MIN(tileK0, ROUND(CACL_TWO * ROUND(N1, ROUND_16) / CACL_TWO, ROUND_16));
            }
            tileK0 = MIN(tileK0, CACL_TWO * ROUND(N1, ROUND_16));
            if (tileK0 > N1_MAX64) {
                tileK0 = ROUND(tileK0, TITLE_CONST);
            }
        } else {
            if (tileK0 > CACL_TWO * N1 / CACL_TWO && tileK0 < CACL_TWO * N1) {
                tileK0 = MIN(tileK0, ROUND(CACL_TWO * N1 / CACL_TWO, ROUND_16));
            }
            tileK0 = MIN(tileK0, ROUND(CACL_TWO * N1, ROUND_16));
        }
        if (tileM0 > CACL_TWO * N1 / CACL_TWO && tileM0 < CACL_TWO * N1) {
            tileM0 = MIN(tileM0, ROUND(CACL_TWO * N1 / CACL_TWO, ROUND_16));
        }
        tileM0 = MIN(tileM0, ROUND(CACL_TWO * N1, ROUND_16));
        tileN0 = MIN(tileN0, ROUND(N2, ROUND_16));
    }
    if (((stepLen <= STEP_LEN_THREE || stepIndex != stepLen - CACL_TWO)) && tileN0 > N1_MAX64) {
        if (tileK0 * ROUND(tileN0, TITLE_CONST) <= L0AB_PINGPONG_BUFFER_LEN * CACL_TWO) {
            tileN0 = ROUND(tileN0, TITLE_CONST);
        } else {
            tileN0 = tileN0 / TITLE_CONST * TITLE_CONST;
        }
    }
}

bool checkSizeToMalloc(size_t size)
{
    if (size == 0) {
        return false;
    }

    return true;
}
