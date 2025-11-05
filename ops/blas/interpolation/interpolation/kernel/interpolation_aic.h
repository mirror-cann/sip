/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INTERPOLATION_AIC_H
#define INTERPOLATION_AIC_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "interpolation_const.h"

using namespace AscendC;
using namespace matmul;

namespace Interpolation {

template <typename T>
class InterpolationAIC {
public:
    __aicore__ inline InterpolationAIC(){};
    __aicore__ inline void Init(InterpolationKernelParam kernelParam, GM_ADDR workspace,
                                AsdSip::InterpolationTilingData *tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ComputeByLine(uint64_t offset, uint64_t dataCount, uint64_t inputOffset);
    __aicore__ inline void Compute(uint64_t offset, uint64_t dataCount, uint64_t inputOffset);

    using aType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using bType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using cType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    MatmulImpl<aType, bType, cType, biasType> mm;

private:
    uint64_t baseBlockIdx;

    // tiling params
    int64_t interpLength;
    int64_t signalLength;
    int64_t batch;
    int64_t quantNum;
    int64_t interpNum;

    int64_t workspaceSize;
    int64_t usedCubeCoreNum;
    int64_t blocksPerLine;
    int64_t tailLines;
    int64_t linesPerCore;
    int64_t numPerBlock;

    int64_t maxCount = MAX_COUNT;

private:
    __gm__ T *__restrict__ inputGM;
    __gm__ T *__restrict__ workspaceGM;
    __gm__ T *__restrict__ outGM;

    GlobalTensor<T> matA;  // 系数矩阵
    GlobalTensor<T> matB;  // 插值关联点
    GlobalTensor<T> matC;
};

template <typename T>
__aicore__ inline void InterpolationAIC<T>::Init(InterpolationKernelParam kernelParam, GM_ADDR workspace,
                                                 AsdSip::InterpolationTilingData *tilingData)
{
    baseBlockIdx = GetBlockIdx();

    interpNum = tilingData->tabInterpNum;
    quantNum = tilingData->tabQuantNum;
    batch = tilingData->batch;
    signalLength = tilingData->signalLength;
    interpLength = tilingData->interpLength;

    numPerBlock = tilingData->numPerBlock;
    linesPerCore = tilingData->linesPerCore;
    tailLines = tilingData->tailLines;
    blocksPerLine = tilingData->blocksPerLine;
    if (blocksPerLine == 0) {
        blocksPerLine = 1;
    }
    usedCubeCoreNum = tilingData->usedCubeCoreNum;
    if (usedCubeCoreNum == 0) {
        usedCubeCoreNum = 1;
    }
    workspaceSize = tilingData->workspaceSize;

    inputGM = (__gm__ T *__restrict__)(kernelParam.x);
    workspaceGM = (__gm__ T *__restrict__)workspace + workspaceSize * baseBlockIdx * 2;
    outGM = (__gm__ float *__restrict__)(kernelParam.out);
}

template <typename T>
__aicore__ inline void InterpolationAIC<T>::Process()
{
    uint64_t inputOffset = (baseBlockIdx * signalLength * linesPerCore) * 2;
    uint64_t posOffset = baseBlockIdx * interpLength * linesPerCore;
    for (int i = 0; i < linesPerCore; i++) {
        ComputeByLine(posOffset, interpLength, inputOffset);
        inputOffset += signalLength * 2;
        posOffset += interpLength;
    }

    if (0 != tailLines) {
        int lineOffset = 0;
        int blockOffset = 0;
        int blocks = blocksPerLine * tailLines;
        int tailblocks = blocks % usedCubeCoreNum;
        int blocksPerCore = blocks / usedCubeCoreNum;
        if ((baseBlockIdx > tailblocks) || (baseBlockIdx == tailblocks)) {
            blockOffset = (blocksPerCore * baseBlockIdx + tailblocks) % blocksPerLine;
            lineOffset = (blocksPerCore * baseBlockIdx + tailblocks) / blocksPerLine;
        } else {
            blocksPerCore += 1;
            blockOffset = blocksPerCore * baseBlockIdx % blocksPerLine;
            lineOffset = blocksPerCore * baseBlockIdx / blocksPerLine;
        }
        if (0 == blocksPerCore) {
            return;
        }

        inputOffset = (lineOffset + linesPerCore * usedCubeCoreNum) * signalLength * 2;
        posOffset = (lineOffset + linesPerCore * usedCubeCoreNum) * interpLength + blockOffset * numPerBlock;
        int dataCount = numPerBlock * blocksPerCore;
        if (blocksPerLine > (blockOffset + blocksPerCore)) {
            ComputeByLine(posOffset, dataCount, inputOffset);
        } else {
            dataCount = interpLength - numPerBlock * blockOffset;
            ComputeByLine(posOffset, dataCount, inputOffset);

            if (blocksPerLine < (blockOffset + blocksPerCore)) {  // 换行
                posOffset = (lineOffset + linesPerCore * usedCubeCoreNum + 1) * interpLength;
                inputOffset += 2 * signalLength;
                dataCount = (blocksPerCore + blockOffset - blocksPerLine) * numPerBlock;
                ComputeByLine(posOffset, dataCount, inputOffset);
            }
        }
    }
}

template <typename T>
__aicore__ inline void InterpolationAIC<T>::Compute(uint64_t offset, uint64_t dataCount, uint64_t inputOffset)
{
    int32_t length = numPerBlock;  // 每次信号处理长度
    int32_t times = dataCount / length;
    for (int32_t i = 0; i < times; i++) {
        if (offset != baseBlockIdx * linesPerCore * interpLength - maxCount || i != times - 1) {
            wait_flag_dev(AIV_FINISH_FLAG_ID);
        }

        __gm__ T *__restrict__ workspaceGM1 = workspaceGM + (i % 2) * workspaceSize;
        GlobalTensor<uint32_t> matInfoGM;
        matInfoGM.SetGlobalBuffer((__gm__ uint32_t *)workspaceGM1, 16);
        uint32_t matSizeM = matInfoGM.GetValue(0);  // matSizeM : length * 2
        uint32_t matSizeN = matInfoGM.GetValue(1);
        uint32_t matSizeK = matInfoGM.GetValue(2);  // matSizeK : len * 2
        uint32_t begin = matInfoGM.GetValue(3);
        uint32_t end = matInfoGM.GetValue(4);

        // workspaceSize : 16 + (len + 8) * 2 + (len * 2) * (length * 2)
        matA.SetGlobalBuffer(workspaceGM1 + 16 + matSizeK + 16, matSizeM * matSizeK);
        if (begin < interpNum / 2 - 1) {
            matB.SetGlobalBuffer(workspaceGM1 + 16 + (begin + 1) * 2, matSizeK * matSizeN);
        } else if (end + interpNum / 2 > signalLength - 1) {
            matB.SetGlobalBuffer(workspaceGM1 + 16 + (end + interpNum / 2 - signalLength + 1) * 2, matSizeK * matSizeN);
        } else {
            matB.SetGlobalBuffer(inputGM + inputOffset + (begin - interpNum / 2 + 1) * 2, matSizeK * matSizeN);
        }
        matC.SetGlobalBuffer(outGM + (offset + length * i) * 2, matSizeM * matSizeN);

        mm.SetSingleShape(matSizeM, matSizeN, matSizeK);
        mm.SetOrgShape(matSizeM, matSizeN, matSizeK);
        mm.SetTensorA(matA);
        mm.SetTensorB(matB);
        mm.IterateAll(matC);
        mm.End();

        DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(matInfoGM);
        ffts_cross_core_sync(PIPE_FIX, 0x1 | (SYNC_MODE_GROUP << 4) | (AIC_FINISH_FLAG_ID << 8));
    }
}

template <typename T>
__aicore__ inline void InterpolationAIC<T>::ComputeByLine(uint64_t posOffset, uint64_t dataLength, uint64_t inputOffset)
{
    int dataCnt = dataLength > maxCount ? maxCount : dataLength;
    int times = dataLength / dataCnt;
    int tail = dataLength % dataCnt;
    for (int i = 0; i < times; i++) {
        Compute(posOffset, dataCnt, inputOffset);
        posOffset += dataCnt;
    }
    if (tail != 0) {
        Compute(posOffset, tail, inputOffset);
    }
}

}

#endif  // INTERPOLATION_AIC_H
