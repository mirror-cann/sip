/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FFT_CORESELECTOR_H__
#define __FFT_CORESELECTOR_H__

#include <cstdint>
#include <vector>
#include <optional>

#include "fftcore/_fft_core_type.h"

// define Range and Enum
namespace SelectCore {

class Range {
public:
    Range(uint32_t lower, uint32_t upper);
    uint32_t GetLower() const;
    uint32_t GetUpper() const;
    static bool Intersect(const Range &one, const Range &other);

private:
    uint32_t lower_;
    uint32_t upper_;
};

class RangeComparator {
public:
    bool operator()(const Range &rangeA, const Range &rangeB) const;
};

class Enum {
public:
    Enum(std::initializer_list<uint32_t> list);
    uint32_t Get(std::vector<uint32_t>::size_type idx) const;
    size_t Size() const;
    bool Contains(uint32_t item) const;

private:
    std::vector<uint32_t> eles_;
};

class EnumComparator {
public:
    bool operator()(const Enum &enumA, const Enum &enumB) const;
};

}

namespace SelectCore {
using EnumCorePair = std::pair<Enum, FFTCoreType>;
using RangeVectorPair = std::pair<Range, std::vector<EnumCorePair>>;
}

// define sort and select utils
namespace SelectCore {

template <typename T>
class RangePairComparator {
public:
    bool operator()(const std::pair<Range, T> &rangePairA, const std::pair<Range, T> &rangePairB) const;
};

template <typename T>
class EnumPairComparator {
public:
    bool operator()(const std::pair<Enum, T> &enumPairA, const std::pair<Enum, T> &enumPairB) const;
};

template <typename T>
using RangePairVector = std::vector<std::pair<Range, T>>;

template <typename T>
using RangePairIter = typename RangePairVector<T>::iterator;

template <typename T>
using EnumPairVector = std::vector<std::pair<Enum, T>>;

template <typename T>
using EnumPairIter = typename EnumPairVector<T>::iterator;

}

// define outer interface to use
namespace SelectCore {

void InitializeConfigs();

std::optional<FFTCoreType> FindConfig(uint32_t batch, uint32_t radix);

}

#endif
