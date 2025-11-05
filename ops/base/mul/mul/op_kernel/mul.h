/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef COMPLEX_CMUL_AIV_H
#define COMPLEX_CMUL_AIV_H

#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

namespace CMul {

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TEMP_BUFFER_NUM = 2;
constexpr uint32_t CAST_BUFFER_NUM = 2;
constexpr uint32_t ELENUM_PER_COMPLEX = 2;
constexpr uint32_t MAX_CORE_COUNT = 48;
constexpr uint32_t FLOAT_PER_BLOCK = 8;
constexpr uint32_t FLOAT_PER_REPEAT = 64;
constexpr uint32_t HALF_PER_BLOCK = 16;

template <typename T>
class CMulAIV {
public:
    __aicore__ inline CMulAIV(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR output, GM_ADDR tilingGm);
    __aicore__ inline void Process();
    __aicore__ inline void ParseTilingData(GM_ADDR tilingGm);
    __aicore__ inline void CopyInGatherMaskData();
    __aicore__ inline void CopyInData(uint64_t offset, uint32_t curCalNum);
    __aicore__ inline void CalC64(uint32_t curCalNum, LocalTensor<float> inXLocal, LocalTensor<float> inYLocal);
    __aicore__ inline void Compute(uint32_t curCalNum);
    __aicore__ inline void CopyOutRes(uint64_t offset, uint32_t curCalNum);

private:
    TPipe pipe;

    GlobalTensor<T> inXGM;
    GlobalTensor<T> inYGM;
    GlobalTensor<T> outGM;

    TQue<QuePosition::VECIN, BUFFER_NUM> inXQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> inYQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    TBuf<TPosition::VECCALC> maskBuf;
    TBuf<TPosition::VECCALC> castBuf;
    TBuf<TPosition::VECCALC> tmpBuf;

    LocalTensor<uint32_t> maskTensor;
    LocalTensor<float> castTensor;
    LocalTensor<float> tmpTensor;

    uint32_t n = 0;  // total num(complex64)
    uint32_t useCoreNum = 0;
    int64_t calNum = 0;
    uint32_t maxDataCnt = 0;
    int64_t startOffset = 0;
    int32_t vecIdx;
    int32_t elementsPerBlock = 0;
};

template <typename T>
__aicore__ inline void CMulAIV<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR output, GM_ADDR tilingGm)
{
    this->vecIdx = GetBlockIdx();
    ParseTilingData(tilingGm);

    if constexpr (IsSameType<T, half>::value) {
        this->elementsPerBlock = HALF_PER_BLOCK;
        this->maxDataCnt = 11 * 1024 / sizeof(T) / ELENUM_PER_COMPLEX;
        this->pipe.InitBuffer(this->castBuf, this->maxDataCnt * ELENUM_PER_COMPLEX * sizeof(float) * CAST_BUFFER_NUM);
        this->castTensor = castBuf.Get<float>();
    } else {
        this->elementsPerBlock = FLOAT_PER_BLOCK;
        this->maxDataCnt = 19 * 1024 / sizeof(T) / ELENUM_PER_COMPLEX;
    }

    this->inXGM.SetGlobalBuffer((__gm__ T *)x, this->n * ELENUM_PER_COMPLEX);
    this->inYGM.SetGlobalBuffer((__gm__ T *)y, this->n * ELENUM_PER_COMPLEX);
    this->outGM.SetGlobalBuffer((__gm__ T *)output, this->n * ELENUM_PER_COMPLEX);

    this->pipe.InitBuffer(this->inXQueue, BUFFER_NUM, this->maxDataCnt * ELENUM_PER_COMPLEX * sizeof(T));
    this->pipe.InitBuffer(this->inYQueue, BUFFER_NUM, this->maxDataCnt * ELENUM_PER_COMPLEX * sizeof(T));
    this->pipe.InitBuffer(this->outQueue, BUFFER_NUM, this->maxDataCnt * ELENUM_PER_COMPLEX * sizeof(T));

    this->pipe.InitBuffer(maskBuf, this->maxDataCnt * ELENUM_PER_COMPLEX * sizeof(uint32_t));
    this->maskTensor = maskBuf.Get<uint32_t>();

    this->pipe.InitBuffer(tmpBuf, this->maxDataCnt * ELENUM_PER_COMPLEX * sizeof(float) * TEMP_BUFFER_NUM);
    this->tmpTensor = tmpBuf.Get<float>();
}

template <typename T>
__aicore__ inline void CMulAIV<T>::ParseTilingData(GM_ADDR tilingGm)
{
    int64_t locId = 0;
    auto tilingBuf = reinterpret_cast<__gm__ uint8_t *>(tilingGm);

    this->n = (*(__gm__ int64_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(int64_t)));
    locId++;
    this->useCoreNum = (*(__gm__ int64_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(int64_t)));
    locId++;
    this->calNum =
        (*(__gm__ int64_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(int64_t) + sizeof(int64_t) * this->vecIdx));
    locId += MAX_CORE_COUNT;
    this->startOffset =
        (*(__gm__ int64_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(int64_t) + sizeof(int64_t) * this->vecIdx));
}

template <typename T>
__aicore__ inline void CMulAIV<T>::CopyInGatherMaskData()
{
    uint32_t k = 0;
    for (int32_t i = 0; i < this->maxDataCnt; i++) {
        this->maskTensor.SetValue(k++, i * sizeof(uint32_t));
        this->maskTensor.SetValue(k++, (i + this->maxDataCnt) * sizeof(uint32_t));
    }
}

template <typename T>
__aicore__ inline void CMulAIV<T>::CopyInData(uint64_t offset, uint32_t curCalNum)
{
    LocalTensor<T> inXLocal = inXQueue.AllocTensor<T>();
    LocalTensor<T> inYLocal = inYQueue.AllocTensor<T>();

    uint32_t blockLen = curCalNum * ELENUM_PER_COMPLEX * sizeof(T);
    uint8_t paddingNum = this->elementsPerBlock - ((curCalNum * ELENUM_PER_COMPLEX) % this->elementsPerBlock);
    if (paddingNum == this->elementsPerBlock)
        paddingNum = 0;

    DataCopyExtParams copyParams{1, blockLen, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

    uint64_t inOffset = (this->startOffset + offset) * ELENUM_PER_COMPLEX;
    DataCopyPad(inXLocal, this->inXGM[inOffset], copyParams, padParams);
    DataCopyPad(inYLocal, this->inYGM[inOffset], copyParams, padParams);
    inXQueue.EnQue<T>(inXLocal);
    inYQueue.EnQue<T>(inYLocal);
}

template <typename T>
__aicore__ inline void CMulAIV<T>::CalC64(uint32_t curCalNum, LocalTensor<float> inXLocal, LocalTensor<float> inYLocal)
{
    uint64_t rsvdCnt = 0;
    uint16_t repeatNum = (curCalNum * ELENUM_PER_COMPLEX - 1) / FLOAT_PER_REPEAT + 1;

    LocalTensor<float> tmpRrLocal = inXLocal[0];
    LocalTensor<float> tmpIiLocal = inXLocal[this->maxDataCnt];
    LocalTensor<float> tmpRiLocal = inYLocal[0];
    LocalTensor<float> tmpIrLocal = inYLocal[this->maxDataCnt];

    LocalTensor<float> tmpXRealLocal = this->tmpTensor[0];
    LocalTensor<float> tmpXImagLocal = this->tmpTensor[this->maxDataCnt];
    LocalTensor<float> tmpYRealLocal = this->tmpTensor[this->maxDataCnt * ELENUM_PER_COMPLEX];
    LocalTensor<float> tmpYImagLocal = this->tmpTensor[this->maxDataCnt * ELENUM_PER_COMPLEX + this->maxDataCnt];

    GatherMask(tmpXRealLocal, inXLocal, 1, false, 0, {1, repeatNum, 8, 8}, rsvdCnt);
    GatherMask(tmpYRealLocal, inYLocal, 1, false, 0, {1, repeatNum, 8, 8}, rsvdCnt);
    GatherMask(tmpYImagLocal, inYLocal, 2, false, 0, {1, repeatNum, 8, 8}, rsvdCnt);
    GatherMask(tmpXImagLocal, inXLocal, 2, false, 0, {1, repeatNum, 8, 8}, rsvdCnt);
    PipeBarrier<PIPE_V>();

    int32_t calCount = ((curCalNum + FLOAT_PER_BLOCK - 1) / FLOAT_PER_BLOCK) * FLOAT_PER_BLOCK;
    Mul(tmpRrLocal, tmpXRealLocal, tmpYRealLocal, calCount);
    Mul(tmpIiLocal, tmpXImagLocal, tmpYImagLocal, calCount);
    Mul(tmpRiLocal, tmpXRealLocal, tmpYImagLocal, calCount);
    Mul(tmpIrLocal, tmpXImagLocal, tmpYRealLocal, calCount);
    PipeBarrier<PIPE_V>();

    Sub(tmpXRealLocal, tmpRrLocal, tmpIiLocal, calCount);  // rr - ii
    Add(tmpXImagLocal, tmpRiLocal, tmpIrLocal, calCount);  // ri + ir
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void CMulAIV<T>::Compute(uint32_t curCalNum)
{
    LocalTensor<T> inXLocal = inXQueue.DeQue<T>();
    LocalTensor<T> inYLocal = inYQueue.DeQue<T>();
    LocalTensor<T> outLocal = outQueue.AllocTensor<T>();

    int32_t gatherCount = ((curCalNum * ELENUM_PER_COMPLEX - 1) / FLOAT_PER_REPEAT + 1) * FLOAT_PER_REPEAT;

    if constexpr (IsSameType<T, half>::value) {
        LocalTensor<float> castXLocal = this->castTensor[0];
        LocalTensor<float> castYLocal = this->castTensor[this->maxDataCnt * ELENUM_PER_COMPLEX];

        Cast(castXLocal, inXLocal, AscendC::RoundMode::CAST_NONE, curCalNum * ELENUM_PER_COMPLEX);
        Cast(castYLocal, inYLocal, AscendC::RoundMode::CAST_NONE, curCalNum * ELENUM_PER_COMPLEX);
        PipeBarrier<PIPE_V>();

        CalC64(curCalNum, castXLocal, castYLocal);
        inXQueue.FreeTensor(inXLocal);
        inYQueue.FreeTensor(inYLocal);

        Gather(castXLocal, this->tmpTensor, this->maskTensor, (uint32_t)0, gatherCount);
        PipeBarrier<PIPE_V>();
        Cast(outLocal, castXLocal, AscendC::RoundMode::CAST_NONE, curCalNum * ELENUM_PER_COMPLEX);
    } else {
        CalC64(curCalNum, inXLocal, inYLocal);
        inXQueue.FreeTensor(inXLocal);
        inYQueue.FreeTensor(inYLocal);

        Gather(outLocal, this->tmpTensor, this->maskTensor, (uint32_t)0, gatherCount);
    }

    outQueue.EnQue<T>(outLocal);
}

template <typename T>
__aicore__ inline void CMulAIV<T>::CopyOutRes(uint64_t offset, uint32_t curCalNum)
{
    LocalTensor<T> outLocal = outQueue.DeQue<T>();
    uint32_t blockLen = curCalNum * ELENUM_PER_COMPLEX * sizeof(T);
    DataCopyExtParams copyParams{1, blockLen, 0, 0, 0};
    uint64_t outOffset = (this->startOffset + offset) * ELENUM_PER_COMPLEX;
    DataCopyPad(this->outGM[outOffset], outLocal, copyParams);
    outQueue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void CMulAIV<T>::Process()
{
    // copy in gather mask
    CopyInGatherMaskData();

    int64_t loopNum = this->calNum / this->maxDataCnt;
    int64_t tailNum = this->calNum % this->maxDataCnt;
    if (tailNum > 0) {
        loopNum++;
    }

    int64_t curOffset = 0;
    for (int64_t loopId = 0; loopId < loopNum; loopId++) {
        uint32_t curCalNum = this->maxDataCnt;
        if ((loopId == loopNum - 1) && tailNum > 0) {
            curCalNum = tailNum;
        }

        CopyInData(curOffset, curCalNum);
        Compute(curCalNum);
        CopyOutRes(curOffset, curCalNum);

        curOffset += this->maxDataCnt;
    }
}
}  // namespace CMul

#endif  // COMPLEX_MUL_AIV_H