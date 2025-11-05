/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_IAMAX_H
#define ASDSIP_PARAMS_IAMAX_H

#include <cstdint>
#include <string>
#include <sstream>

namespace AsdSip {
namespace OpParam {
struct Iamax {
    int64_t n;
    int64_t incx;
    int64_t dtype;
    bool operator==(const Iamax &other) const
    {
        return this->n == other.n && this->incx == other.incx && this->dtype == other.dtype;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: Iamax"
           << ", n:" << n << ", incx:" << incx << ", dtype:" << dtype;
        return ss.str();
    }
};
}  // namespace OpParam
}  // namespace AsdSip

#endif  // ASDSIP_PARAMS_IAMAX_H