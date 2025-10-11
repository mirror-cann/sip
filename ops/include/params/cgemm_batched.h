/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_CGEMMBATCHED_H
#define ASDSIP_PARAMS_CGEMMBATCHED_H

#include <cstdint>
#include <string>
#include <sstream>
#include <complex>

namespace AsdSip {
namespace OpParam {
struct CgemmBatched {
    int64_t m;
    int64_t k;
    int64_t n;
    int64_t batch;

    bool operator==(const CgemmBatched &other) const
    {
        return this->m == other.m && this->k == other.k && this->n == other.n &&
               this->batch == other.batch;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: CgemmBatched"
           << ", m:" << m << ", k:" << k << ", n:" << n
           << ", batch:" << batch;
        return ss.str();
    }
};
}  // namespace OpParam
}  // namespace AsdSip

#endif  // ASDSIP_PARAMS_CGEMMBATCHED_H
