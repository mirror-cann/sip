/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_CGEMV_BATCHED_H
#define ASDSIP_PARAMS_CGEMV_BATCHED_H

#include <cstdint>
#include <string>
#include <sstream>

namespace AsdSip {
namespace OpParam {
struct CgemvBatched {
    int64_t transType;
    int64_t dataType;
    int64_t m;
    int64_t n;
    int64_t batchCount;
    int64_t maxMatNum;

    bool operator==(const CgemvBatched &other) const
    {
        return this->transType == other.transType && this->dataType == other.dataType
                && this->m == other.m && this->n == other.n
                && this->batchCount == other.batchCount && this->maxMatNum == other.maxMatNum;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "transType:" << transType << ", dataType:" << dataType
            << ", m:" << m << ", n:" << n << ", batchCount:" << batchCount
            << ", maxMatNum:" << maxMatNum;
        return ss.str();
    }
};
}  // namespace OpParam
}  // namespace AsdSip

#endif  // ASDSIP_PARAMS_CGEMV_BATCHED_H