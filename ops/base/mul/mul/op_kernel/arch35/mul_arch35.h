/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MUL_ARCH35_H
#define MUL_ARCH35_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

namespace MulArch35 {
using namespace AscendC;

constexpr uint32_t VF_MAX_THREAD_NUM = 256;

template <typename T>
__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) __aicore__ void MulArch35VfKernel(
    __gm__ T * __restrict__ gm_x,
    __gm__ T * __restrict__ gm_y,
    __gm__ T * __restrict__ gm_output,
    int64_t startOffset,
    int64_t calNum)
{
    int64_t tid = static_cast<int64_t>(Simt::GetThreadIdx<0>());
    int64_t stride = static_cast<int64_t>(Simt::GetThreadNum<0>());
    int64_t startIdx = startOffset;

    for (int64_t idx = tid; idx < calNum; idx += stride) {
        int64_t elemIdx = (startIdx + idx) * 2;
        
        float xr = static_cast<float>(gm_x[elemIdx]);
        float xi = static_cast<float>(gm_x[elemIdx + 1]);
        float yr = static_cast<float>(gm_y[elemIdx]);
        float yi = static_cast<float>(gm_y[elemIdx + 1]);

        float outr = xr * yr - xi * yi;
        float outi = xr * yi + xi * yr;

        gm_output[elemIdx] = static_cast<T>(outr);
        gm_output[elemIdx + 1] = static_cast<T>(outi);
    }
}

template <typename T>
class MulArch35Kernel {
public:
    __aicore__ inline MulArch35Kernel() {}

    __aicore__ inline void Init(
        __gm__ T * __restrict__ gm_x,
        __gm__ T * __restrict__ gm_y,
        __gm__ T * __restrict__ gm_output,
        __gm__ uint8_t * __restrict__ gm_tiling);

    __aicore__ inline void Process();

private:
    __gm__ T * __restrict__ gm_x_;
    __gm__ T * __restrict__ gm_y_;
    __gm__ T * __restrict__ gm_output_;

    int64_t n_;

    int64_t blockIdx_;
    int64_t blockNum_;
    int64_t calNum_;
    int64_t startOffset_;
};

template <typename T>
__aicore__ inline void MulArch35Kernel<T>::Init(
    __gm__ T * __restrict__ gm_x,
    __gm__ T * __restrict__ gm_y,
    __gm__ T * __restrict__ gm_output,
    __gm__ uint8_t * __restrict__ gm_tiling)
{
    blockIdx_ = static_cast<int64_t>(GetBlockIdx());
    blockNum_ = static_cast<int64_t>(GetBlockNum());

    n_ = (*(__gm__ int64_t *)(gm_tiling));

    int64_t perCoreNum = n_ / blockNum_;
    int64_t remainNum = n_ % blockNum_;
    if (blockIdx_ < remainNum) {
        startOffset_ = blockIdx_ * (perCoreNum + 1);
        calNum_ = perCoreNum + 1;
    } else {
        startOffset_ = remainNum * (perCoreNum + 1) + (blockIdx_ - remainNum) * perCoreNum;
        calNum_ = perCoreNum;
    }

    gm_x_ = gm_x;
    gm_y_ = gm_y;
    gm_output_ = gm_output;
}

template <typename T>
__aicore__ inline void MulArch35Kernel<T>::Process()
{
    if (blockIdx_ > blockNum_ - 1) return;
    if (calNum_ <= 0) return;

    Simt::VF_CALL<MulArch35VfKernel<T>>(
        Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1},
        gm_x_,
        gm_y_,
        gm_output_,
        startOffset_,
        calNum_);
}

} // namespace MulArch35

#endif // MUL_ARCH35_H