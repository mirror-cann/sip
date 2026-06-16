/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASDSIP_PARAMS_FFT_C2C2D_ARCH35_H
#define ASDSIP_PARAMS_FFT_C2C2D_ARCH35_H

#include <cstdint>
namespace AsdSip {
namespace OpParam {

enum class FftC2C2DArch35Mode : int32_t {
    RADIX2 = 0,
    MIXED_RADIX = 1,
};

}  // namespace OpParam
}  // namespace AsdSip

#endif
