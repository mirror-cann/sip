/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_CTRMV_H
#define ASDSIP_PARAMS_CTRMV_H

#include <cstdint>
#include <string>
#include <sstream>
#include <mki/utils/SVector/SVector.h>

namespace AsdSip {
namespace OpParam {
struct Ctrmv {
    int64_t uplo;
    int64_t trans;
    int64_t diag;
    int64_t n;
    int64_t lda;
    int64_t incx;

    bool operator==(const Ctrmv &other) const
    {
        return this->uplo == other.uplo && this->trans == other.trans && this->diag == other.diag &&
               this->n == other.n && this->lda == other.lda && this->incx == other.incx;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: Ctrmv"
           << ", uplo:" << uplo << ", trans:" << trans << ", diag:" << diag << ", n:" << n << ", lda:" << lda
           << ", incx:" << incx;
        return ss.str();
    }
};
}  // namespace OpParam
}  // namespace AsdSip

#endif  // ASDSIP_PARAMS_CTRMV_H