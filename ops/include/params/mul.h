/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_COMPLEX_MUL_H
#define ASDSIP_PARAMS_COMPLEX_MUL_H

#include <cstdint>
#include <string>
#include <sstream>
#include <mki/utils/SVector/SVector.h>

namespace AsdSip {
namespace OpParam {
struct CMul {
    // the number of complex elements
    int64_t n;

    enum class CMulType {
        MUL_C64 = 0,
        MUL_C32,
    };

    CMulType cMulType;

    bool operator==(const CMul &other) const
    {
        bool nEqual = this->n == other.n;
        bool typeEqual = this->cMulType == other.cMulType;
        return nEqual && typeEqual;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: CMul" << ", n:" << n << ", CMulType:" << static_cast<int>(cMulType);
        return ss.str();
    }
};

}  // namespace OpParam
}  // namespace AsdSip

#endif  // ASDSIP_PARAMS_COMPLEX_MUL_H