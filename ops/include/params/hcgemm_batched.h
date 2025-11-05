/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_HCGEMMBATCHED_H
#define ASDSIP_PARAMS_HCGEMMBATCHED_H

#include <cstdint>
#include <string>
#include <sstream>
#include <complex>

namespace AsdSip {
namespace OpParam {
struct HCgemmBatched {
    int64_t m;
    int64_t k;
    int64_t n;
    int64_t batch;

    bool operator==(const HCgemmBatched &other) const
    {
        return this->m == other.m && this->k == other.k && this->n == other.n &&
               this->batch == other.batch;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: HCgemmBatched"
           << ", m:" << m << ", k:" << k << ", n:" << n
           << ", batch:" << batch;
        return ss.str();
    }
};
}  // namespace OpParam
}  // namespace AsdSip

#endif  // ASDSIP_PARAMS_HCGEMMBATCHED_H