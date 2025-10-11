/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_CGEMV_H
#define ASDSIP_PARAMS_CGEMV_H

#include <cstdint>
#include <string>
#include <sstream>
#include <complex>

namespace AsdSip {
namespace OpParam {
struct Cgemv {
    int64_t transA;
    int64_t M;
    int64_t N;

    int64_t lda;
    int64_t incx;
    int64_t incy;

    std::complex<float> alpha;
    std::complex<float> beta;

    bool operator==(const Cgemv &other) const
    {
        return this->transA == other.transA && this->M == other.M && this->N == other.N && this->lda == other.lda &&
               this->incx == other.incx && this->incy == other.incy && this->alpha == other.alpha &&
               this->beta == other.beta;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "transA:" << transA << ", M:" << M << ", N:" << N << ", lda:" << lda << ", incx:" << incx
           << ", incy:" << incy << ", alpha:" << alpha << ", beta:" << beta;
        return ss.str();
    }
};
}  // namespace OpParam
}  // namespace AsdSip

#endif  // ASDSIP_PARAMS_CGEMV_H