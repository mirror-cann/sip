/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_STRMV_H
#define ASDSIP_PARAMS_STRMV_H

#include <cstdint>
#include <string>
#include <sstream>

namespace AsdSip {
namespace OpParam {
struct Strmv {
    uint32_t uplo;   // 'u' or 'U' -> 1;'l' or 'L' -> 0;
    uint32_t trans;  // 'N' or 'n'-> 0; 't','c'.'T','C' -> 1
    uint32_t diag;   // 'u', 'U' -> 1; 'N', 'n' -> 0
    uint32_t lda;    // 行间隔
    uint32_t incx;   // 相邻元素

    bool operator==(const Strmv &other) const
    {
        return this->diag == other.diag && this->uplo == other.uplo && this->trans == other.trans;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: Strmv"
           << ", diag:" << diag << ", uplo:" << uplo << ", trans" << trans << ", lda" << lda << ", incx" << incx;
        return ss.str();
    }
};
}  // namespace OpParam
}  // namespace AsdSip

#endif  // ASDSIP_PARAMS_STRMV_H