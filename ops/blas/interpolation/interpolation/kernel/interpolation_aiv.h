/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INTERPOLATION_AIV_H
#define INTERPOLATION_AIV_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "interpolation_const.h"

using namespace AscendC;
using namespace matmul;

namespace Interpolation {

template <typename T>
class InterpolationAIV {
public:
    __aicore__ inline InterpolationAIV(){};
    __aicore__ inline void Init(InterpolationKernelParam kernelParam, GM_ADDR workspace,
                                AsdSip::InterpolationTilingData *tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ComputeByLine(uint64_t offset, uint64_t dataCount, uint64_t inputOffset);
    __aicore__ inline void Compute(uint64_t offset, uint64_t dataCount, uint64_t inputOffset);

private:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b)
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    };

private:
    static constexpr uint32_t NUM_FOUR = 4;
    static constexpr uint32_t BLOCK_NUM = 8;  // 32/sizeof(float)
    static constexpr uint32_t TAB_OFFSET = (33 * 2) * (16 * 2 + 8);

    // tiling params
    int64_t interpNum;
    int64_t quantNum;
    int64_t batch;
    int64_t signalLength;
    int64_t interpLength;

    int64_t numPerBlock;
    int64_t linesPerCore;
    int64_t tailLines;
    int64_t blocksPerLine;
    int64_t usedCubeCoreNum;
    int64_t workspaceSize;

    uint64_t blockIdx;
    uint64_t subblockIdx;
    uint64_t baseBlockIdx;

    int64_t tabSize;
    int64_t maxCount = MAX_COUNT;
    int64_t sign = 0;

private:
    __gm__ float *__restrict__ inputGM;
    __gm__ float *__restrict__ tabGM;
    __gm__ int32_t *__restrict__ positionGM;
    __gm__ float *__restrict__ outGM;
    __gm__ float *__restrict__ workspaceGM;
    __gm__ int16_t *__restrict__ tabIndexGM;

    __ubuf__ float *tabUB = reinterpret_cast<__ubuf__ float *>((uintptr_t)0);
    __ubuf__ float *posUB;
    __ubuf__ int32_t *posFloorUB;
    __ubuf__ float *posFracUB;
    __ubuf__ int16_t *tabIndexUB;
    __ubuf__ float *tempUB;
};

template <typename T>
__aicore__ inline void InterpolationAIV<T>::Init(InterpolationKernelParam kernelParam, GM_ADDR workspace,
                                                 AsdSip::InterpolationTilingData *tilingData)
{
    blockIdx = GetBlockIdx();
    subblockIdx = blockIdx % 2;
    baseBlockIdx = blockIdx / 2;

    interpNum = tilingData->tabInterpNum;
    quantNum = tilingData->tabQuantNum;
    batch = tilingData->batch;
    signalLength = tilingData->signalLength;
    interpLength = tilingData->interpLength;

    numPerBlock = tilingData->numPerBlock;
    linesPerCore = tilingData->linesPerCore;
    tailLines = tilingData->tailLines;
    blocksPerLine = tilingData->blocksPerLine;
    usedCubeCoreNum = tilingData->usedCubeCoreNum;
    workspaceSize = tilingData->workspaceSize;

    inputGM = (__gm__ float *__restrict__)(kernelParam.x);
    tabGM = (__gm__ float *__restrict__)(kernelParam.tab);
    positionGM = (__gm__ int32_t *__restrict__)(kernelParam.pos);
    outGM = (__gm__ float *__restrict__)(kernelParam.out);
    workspaceGM = (__gm__ float *__restrict__)workspace + workspaceSize * baseBlockIdx * 2;
    tabIndexGM = (__gm__ int16_t *__restrict__)(kernelParam.tabIndex);

    tabSize = TAB_OFFSET * NUM_FOUR;

    posUB = reinterpret_cast<__ubuf__ float *>(tabUB + tabSize);
    posFloorUB = reinterpret_cast<__ubuf__ int32_t *>(posUB + maxCount);
    posFracUB = reinterpret_cast<__ubuf__ float *>(posFloorUB + maxCount);
    tabIndexUB = reinterpret_cast<__ubuf__ int16_t *>(posFracUB + maxCount);
    tempUB = reinterpret_cast<__ubuf__ float *>(tabIndexUB);

    copy_gm_to_ubuf(tabUB, tabGM,
                    0,                    // sid 一般0
                    1,                    // number of bursts
                    tabSize / BLOCK_NUM,  // burst_len 32B unit
                    0,                    // src stride 即burst stride
                    0);
}

template <typename T>
__aicore__ inline void InterpolationAIV<T>::Process()
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));

    uint64_t inputOffset = (baseBlockIdx * linesPerCore * signalLength) * 2;
    uint64_t posOffset = baseBlockIdx * linesPerCore * interpLength;
    for (int i = 0; i < linesPerCore; i++) {
        ComputeByLine(posOffset, interpLength, inputOffset);
        posOffset += interpLength;
        inputOffset += signalLength * 2;
    }

    if (tailLines != 0) {
        int blocks = blocksPerLine * tailLines;
        int blocksPerCore = blocks / usedCubeCoreNum;
        int tailblocks = blocks % usedCubeCoreNum;
        int lineOffset = 0;
        int blockOffset = 0;
        if (baseBlockIdx < tailblocks) {
            blocksPerCore += 1;
            lineOffset = blocksPerCore * baseBlockIdx / blocksPerLine;
            blockOffset = blocksPerCore * baseBlockIdx % blocksPerLine;
        } else {
            lineOffset = (blocksPerCore * baseBlockIdx + tailblocks) / blocksPerLine;
            blockOffset = (blocksPerCore * baseBlockIdx + tailblocks) % blocksPerLine;
        }

        if (blocksPerCore == 0) {
            return;
        }

        inputOffset = (linesPerCore * usedCubeCoreNum + lineOffset) * signalLength * 2;
        posOffset = (linesPerCore * usedCubeCoreNum + lineOffset) * interpLength + blockOffset * numPerBlock;
        int dataCount = blocksPerCore * numPerBlock;
        if (blockOffset + blocksPerCore < blocksPerLine) {
            ComputeByLine(posOffset, dataCount, inputOffset);
        } else {
            dataCount = interpLength - blockOffset * numPerBlock;
            ComputeByLine(posOffset, dataCount, inputOffset);

            if (blockOffset + blocksPerCore > blocksPerLine) {  // 换行
                posOffset = (linesPerCore * usedCubeCoreNum + lineOffset + 1) * interpLength;
                inputOffset += signalLength * 2;
                dataCount = (blockOffset + blocksPerCore - blocksPerLine) * numPerBlock;
                ComputeByLine(posOffset, dataCount, inputOffset);
            }
        }
    }
}

template <typename T>
__aicore__ inline void InterpolationAIV<T>::ComputeByLine(uint64_t posOffset, uint64_t dataLength, uint64_t inputOffset)
{
    int dataCount = dataLength > maxCount ? maxCount : dataLength;
    int times = dataLength / dataCount;
    int remain = dataLength % dataCount;
    for (int j = 0; j < times; j++) {
        Compute(posOffset, dataCount, inputOffset);
        posOffset += dataCount;
    }
    if (remain != 0) {
        Compute(posOffset, remain, inputOffset);
    }
}

template <typename T>
__aicore__ inline void InterpolationAIV<T>::Compute(uint64_t offset, uint64_t dataCount, uint64_t inputOffset)
{
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    copy_gm_to_ubuf(posFloorUB, positionGM + offset,
                    0,                      // sid 一般0
                    1,                      // number of bursts
                    dataCount / BLOCK_NUM,  // burst_len 32B unit
                    0,                      // src stride 即burst stride
                    0);
    copy_gm_to_ubuf(tabIndexUB, tabIndexGM + offset,
                    0,                      // sid 一般0
                    1,                      // number of bursts
                    dataCount / BLOCK_NUM,  // burst_len 32B unit
                    0,                      // src stride 即burst stride
                    0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

    int32_t length = numPerBlock;  // 每次信号处理长度
    int32_t times = dataCount / length;
    int32_t outOffset = workspaceSize * baseBlockIdx;
    int32_t outOffset1 = 0;

    for (int32_t i = 0; i < times; i++) {
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);

        int32_t begin = posFloorUB[i * length];
        int32_t end = posFloorUB[i * length + length - 1];
        int32_t len = ((end - begin + 1 + (interpNum - 1)) + 3) / 4 * 4;  // Complex 32字节对齐
        int32_t repeat = (len * 2) * (length * 2) / 64;

        __ubuf__ uint32_t *matInfo = reinterpret_cast<__ubuf__ uint32_t *>(tabIndexUB + maxCount);  // M N K
        __ubuf__ float *matB = reinterpret_cast<__ubuf__ float *>(matInfo + 16);
        __ubuf__ float *matTab = reinterpret_cast<__ubuf__ float *>(matB + (len + 8) * 2);

        if (subblockIdx == 0) {
            if (begin < length / 2 - 1) {
                copy_gm_to_ubuf(matB + 8 * 2, inputGM + inputOffset,
                                0,                    // sid 一般0
                                1,                    // number of bursts
                                len * 2 / BLOCK_NUM,  // burst_len 32B unit
                                0,                    // src stride 即burst stride
                                0);
                set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                float beginReal = matB[8 * 2];
                float beginImg = matB[8 * 2 + 1];
                for (int j = 0; j < 8; j++) {
                    matB[2 * j] = beginReal;
                    matB[2 * j + 1] = beginImg;
                }
            }

            if (end + length / 2 > signalLength - 1) {
                set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
                copy_gm_to_ubuf(matB, inputGM + inputOffset + (signalLength - len) * 2,
                                0,                    // sid 一般0
                                1,                    // number of bursts
                                len * 2 / BLOCK_NUM,  // burst_len 32B unit
                                0,                    // src stride 即burst stride
                                0);
                set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                float beginReal = matB[len * 2 - 2];
                float beginImg = matB[len * 2 - 1];
                for (int j = 0; j < 8; j++) {
                    matB[2 * (len + j)] = beginReal;
                    matB[2 * (len + j) + 1] = beginImg;
                }
            }
            matInfo[0] = length * 2;  // M
            matInfo[1] = 1;           // N
            matInfo[2] = len * 2;     // K
            matInfo[3] = begin;
            matInfo[4] = end;
        }

        int32_t repeatTimes = repeat / 255;
        int32_t repeatRemain = repeat % 255;
        for (int i = 0; i < repeatTimes; i++) {
            vector_dup(matTab + i * 64 * 255,  // dst
                       (float)0,               // scalar
                       255,                    // repeat  fp32 每次是64数
                       1,                      // dstBlockStride
                       1,                      // srcBlockStride
                       8,                      // dstRepeatStride
                       8                       // srcRepeatStride
            );
        }
        if (repeatRemain != 0) {
            vector_dup(matTab + repeatTimes * 64 * 255,  // dst
                       (float)0,                         // scalar
                       repeatRemain,                     // repeat  fp32 每次是64数
                       1,                                // dstBlockStride
                       1,                                // srcBlockStride
                       8,                                // dstRepeatStride
                       8                                 // srcRepeatStride
            );
        }
        pipe_barrier(PIPE_V);

        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        for (int32_t j = subblockIdx * length / 2; j < subblockIdx * length / 2 + length / 2; j++) {
            int32_t tab_index = tabIndexUB[i * length + j];  // 每个点的系数行号
            int32_t zeroFrontLen = posFloorUB[i * length + j] - posFloorUB[i * length];
            if (end + length / 2 > signalLength - 1) {
                zeroFrontLen = zeroFrontLen + len - (end - begin + 1 + (interpNum - 1));
            }
            if ((zeroFrontLen % 4) == 0) {
                copy_ubuf_to_ubuf(matTab + j * 2 * len * 2 + (zeroFrontLen - zeroFrontLen % 4) * 2,
                                  tabUB + (zeroFrontLen % 4) * TAB_OFFSET + (tab_index * 2) * (16 * 2 + 8), 0,
                                  (16 * 2) / BLOCK_NUM, 1, 0, 0);
                copy_ubuf_to_ubuf(matTab + (j * 2 + 1) * len * 2 + (zeroFrontLen - zeroFrontLen % 4) * 2,
                                  tabUB + (zeroFrontLen % 4) * TAB_OFFSET + (tab_index * 2 + 1) * (16 * 2 + 8), 0,
                                  (16 * 2) / BLOCK_NUM, 1, 0, 0);
            } else {
                copy_ubuf_to_ubuf(matTab + j * 2 * len * 2 + (zeroFrontLen - zeroFrontLen % 4) * 2,
                                  tabUB + (zeroFrontLen % 4) * TAB_OFFSET + (tab_index * 2) * (16 * 2 + 8), 0,
                                  (16 * 2 + 8) / BLOCK_NUM, 1, 0, 0);
                copy_ubuf_to_ubuf(matTab + (j * 2 + 1) * len * 2 + (zeroFrontLen - zeroFrontLen % 4) * 2,
                                  tabUB + (zeroFrontLen % 4) * TAB_OFFSET + (tab_index * 2 + 1) * (16 * 2 + 8), 0,
                                  (16 * 2 + 8) / BLOCK_NUM, 1, 0, 0);
            }
        }
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        if (subblockIdx == 0) {
            copy_ubuf_to_gm(workspaceGM + (i % 2) * workspaceSize, matInfo, 0, 1,
                            (16 + (len * 2) * (length * 2) / 2 + (len + 8) * 2) / BLOCK_NUM, 0, 0);
        } else {
            copy_ubuf_to_gm(workspaceGM + (i % 2) * workspaceSize + (len * 2) * (length * 2) / 2 + 16 + (len + 8) * 2,
                            matTab + (len * 2) * (length * 2) / 2, 0, 1, (len * 2) * (length * 2) / 2 / BLOCK_NUM, 0,
                            0);
        }

        ffts_cross_core_sync(PIPE_MTE3, 0x1 | (SYNC_MODE_GROUP << 4) | (AIV_FINISH_FLAG_ID << 8));
        if (sign == 0) {
            sign = 1;
            continue;
        }
        wait_flag_dev(AIC_FINISH_FLAG_ID);
    }
}

}
#endif  // INTERPOLATION_AIV_H