/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>

#include "log/log.h"


#include "fftcore/select_core.h"

namespace SelectCore {

inline Range::Range(uint32_t lower, uint32_t upper) : lower_(lower), upper_(upper) {}

inline uint32_t Range::GetLower() const
{
    return lower_;
}

inline uint32_t Range::GetUpper() const
{
    return upper_;
}

inline bool Range::Intersect(const Range &one, const Range &other)
{
    if (one.lower_ >= other.upper_ || other.lower_ >= one.upper_) {
        return false;
    }

    return true;
}

inline bool RangeComparator::operator()(const Range &rangeA, const Range &rangeB) const
{
    return rangeA.GetLower() < rangeB.GetLower();
}

static RangeComparator rangeComparator;

inline Enum::Enum(std::initializer_list<uint32_t> list) : eles_{list}
{
    if (eles_.size() == 0) {
        ASDSIP_LOG(ERROR) << "input is empty.";
        throw std::runtime_error("input is empty.");
    }
}

inline uint32_t Enum::Get(std::vector<uint32_t>::size_type idx) const
{
    return eles_.at(idx);
};

inline size_t Enum::Size() const
{
    return eles_.size();
};

inline bool Enum::Contains(uint32_t item) const
{
    return std::find(eles_.begin(), eles_.end(), item) != eles_.end();
}

inline bool EnumComparator::operator()(const Enum &enumA, const Enum &enumB) const
{
    return enumA.Get(0) < enumB.Get(0);
}

static EnumComparator enumComparator;

}

namespace SelectCore {

template <typename T>
inline bool RangePairComparator<T>::operator()(const std::pair<Range, T> &rangePairA,
                                               const std::pair<Range, T> &rangePairB) const
{
    return rangeComparator(rangePairA.first, rangePairB.first);
}

static RangePairComparator<std::vector<EnumCorePair>> rangeVectorComparator;

template <typename T>
inline bool EnumPairComparator<T>::operator()(const std::pair<Enum, T> &enumPairA,
                                              const std::pair<Enum, T> &enumPairB) const
{
    return enumComparator(enumPairA.first, enumPairB.first);
}

static EnumPairComparator<FFTCoreType> enumCoreComparator;

template <typename T>
RangePairIter<T> SelectRangePair(RangePairIter<T> first, RangePairIter<T> last, uint32_t val)
{
    RangePairIter<T> end = last;
    RangePairIter<T> mid;

    while (first < last) {
        int64_t numElements = last - first;
        mid = first + ((static_cast<uint64_t>(numElements) - 1) >> 1);

        uint32_t rangeLower = mid->first.GetLower();
        uint32_t rangeUpper = mid->first.GetUpper();
        if (val < rangeLower) {
            last = mid;
        } else if (val >= rangeUpper) {
            first = mid + 1;
        } else {
            return mid;
        }
    }

    return end;
}

template <typename T>
bool CheckIntersections(RangePairIter<T> first, RangePairIter<T> last)
{
    RangePairIter<T> prevIter = first;
    for (RangePairIter<T> iter = first + 1; iter < last; iter++) {
        if (Range::Intersect(prevIter->first, iter->first)) {
            return true;
        }

        prevIter = iter;
    }

    return false;
}

template <typename T>
EnumPairIter<T> SelectEnumPair(EnumPairIter<T> first, EnumPairIter<T> last, uint32_t val)
{
    for (; first < last; first++) {
        if (first->first.Contains(val)) {
            return first;
        }
    }

    return last;
}

}

namespace SelectCore {

static std::vector<RangeVectorPair> configs = {
    {{8, 96},
     {{{256, 512, 1024}, FFTCoreType::kDft},
      {{2048, 4096, 8192, 16384}, FFTCoreType::kFftB},
      {{32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304}, FFTCoreType::kFftN}}},
    {{96, 1024}, {{{256, 512}, FFTCoreType::kDft}, {{1024, 2048, 4096, 8192, 16384}, FFTCoreType::kFftB}}},
    {{1024, 65537}, {{{256}, FFTCoreType::kDft}, {{512, 1024, 2048, 4096, 8192, 16384}, FFTCoreType::kFftB}}}};

static bool g_initialized = false;

void InitializeConfigs()
{
    if (g_initialized) {
        return;
    }

    g_initialized = true;

    for (RangeVectorPair &enumCorePairVector : configs) {
        std::sort(enumCorePairVector.second.begin(), enumCorePairVector.second.end(), enumCoreComparator);
    }

    std::sort(configs.begin(), configs.end(), rangeVectorComparator);

    if (CheckIntersections<std::vector<EnumCorePair>>(configs.begin(), configs.end())) {
        ASDSIP_LOG(ERROR) << "there are intersections between input ranges.";
        throw std::runtime_error("there are intersections between input ranges.");
    }
}

std::optional<FFTCoreType> FindConfig(uint32_t batch, uint32_t radix)
{
    auto rangeVectorPairIter = SelectRangePair<std::vector<EnumCorePair>>(configs.begin(), configs.end(), batch);
    if (rangeVectorPairIter == configs.end()) {
        return std::nullopt;
    }

    auto enumPairVector = rangeVectorPairIter->second;

    auto enumCorePairIter = SelectEnumPair<FFTCoreType>(enumPairVector.begin(), enumPairVector.end(), radix);
    if (enumCorePairIter == enumPairVector.end()) {
        return std::nullopt;
    }

    return enumCorePairIter->second;
}

}
