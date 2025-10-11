/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_INTERPOLATION_H
#define ASDSIP_PARAMS_INTERPOLATION_H

#include <cstdint>
#include <string>
#include <sstream>
#include <complex>

namespace AsdSip {
namespace OpParam {
struct Interpolation {
    int32_t tabInterpNum;  // 插值点数
    int32_t tabQuantNum;   // 量化系数
    int32_t batch;         // 信号batch
    int32_t signalLength;  // 信号长度
    int32_t interpLength;  // 插值长度

    bool operator==(const Interpolation &other) const
    {
        return this->tabInterpNum == other.tabInterpNum && this->tabQuantNum == other.tabQuantNum &&
               this->batch == other.batch && this->signalLength == other.signalLength &&
               this->interpLength == other.interpLength;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: Interpolation"
           << ", tabInterpNum:" << tabInterpNum << ", tabQuantNum:" << tabQuantNum << ", batch:" << batch
           << ", signalLength:" << signalLength << ", interpLength:" << interpLength;
        return ss.str();
    }
};
}  // namespace OpParam
}  // namespace AsdSip

#endif  // ASDSIP_PARAMS_INTERPOLATION_H