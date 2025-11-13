/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_STRMM_H
#define ASDSIP_PARAMS_STRMM_H

#include <cstdint>
#include <string>
#include <sstream>
#include <mki/utils/SVector/SVector.h>

namespace AsdSip {
namespace OpParam {

constexpr float FLOAT_THRESHOLD = 1e-9;

struct Strmm {
    uint32_t side;
    uint32_t uplo;
    uint32_t transLeft;
    uint32_t transRight;
    uint32_t diag;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t lessFlag;
    float alpha;

    bool operator==(const Strmm &other) const
    {
        bool isAlphaEqual = abs(this->alpha - other.alpha) < FLOAT_THRESHOLD;
        return this->side == other.side && this->uplo == other.uplo && this->transLeft == other.transLeft &&
               this->transRight == other.transRight && this->diag == other.diag && this->m == other.m &&
               this->n == other.n && isAlphaEqual;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: Strmm"
           << ", alpha:" << alpha << ", side:" << side << ", uplo:" << uplo << ", transLeft:" << transLeft
           << ", transRight:" << transRight << ", diag:" << diag << ", m:" << m << ", n:" << n;
        return ss.str();
    }
};
}  // namespace OpParam
}  // namespace AsdSip

#endif  // ASDSIP_PARAMS_STRMM_H