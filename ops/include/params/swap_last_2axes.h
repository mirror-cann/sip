/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_SWAP_LAST_2AXES_H
#define ASDSIP_PARAMS_SWAP_LAST_2AXES_H

#include <cstdint>
#include <string>
#include <sstream>
#include <mki/utils/SVector/SVector.h>

namespace AsdSip {
namespace OpParam {
struct SwapLast2Axes {
    uint32_t batch;
    uint32_t row;
    uint32_t col;

    bool operator==(const SwapLast2Axes &other) const
    {
        return this->batch == other.batch && this->row == other.row && this->col == other.col;
    };

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: swap_last_2axes "
           << ", batch:" << batch << ", row:" << row << ", col:" << col;
        return ss.str();
    }
};
}  // namespace OpParam
}  // namespace AsdSip

#endif  // ASDSIP_PARAMS_SWAP_LAST_2AXES_H
