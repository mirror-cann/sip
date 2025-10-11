/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_INTERP_BY_COEFF_H
#define ASDSIP_PARAMS_INTERP_BY_COEFF_H

#include <cstdint>
#include <string>
#include <sstream>
#include <complex>

namespace AsdSip {
namespace OpParam {
struct InterpByCoeff {
    int32_t batch;            // 信号波束
    int32_t rsNum;            // 参考信号数
    int32_t totalSubcarrier;  // 总子载波数
    int32_t interpLength;     // 插值长度  14 - rsNum

    bool operator==(const InterpByCoeff &other) const
    {
        return this->batch == other.batch && this->rsNum == other.rsNum &&
               this->totalSubcarrier == other.totalSubcarrier && this->interpLength == other.interpLength;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: Interpolation"
           << ", batch:" << batch << ", rsNum:" << rsNum << ", totalSubcarrier:" << totalSubcarrier
           << ", interpLength:" << interpLength;
        return ss.str();
    }
};
}  // namespace OpParam
}  // namespace AsdSip

#endif  // ASDSIP_PARAMS_INTERP_BY_COEFF_H