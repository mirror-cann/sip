/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "../../../../include/common/common.h"
#include "../../../../include/common/common_func.h"
#include "../../../../include/common/simd.h"
#include "../../../../include/common/iterator.h"
#include "../../../../include/common/mma.h"
#include "../../../../include/common/utils.h"

#include "matmul_base_impl.h"

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

// ----------------- common parameters ---------------------
constexpr uint32_t BASE_COL_BLOCK = 128;

// ----------------- cube parameters ---------------------
#ifdef __DAV_C220_CUBE__
#endif

#ifdef __DAV_C220_VEC__

#endif
// ----------------- parameters end ---------------------

// ----------------- common func ---------------------
__aicore__ inline int64_t GET_FFST_MSG(int64_t mode, int64_t flagId)
{
    int64_t modeOffset = 4;
    int64_t flagOffset = 8;
    return 1 | (mode << modeOffset) | (flagId << flagOffset);
}
// ----------------- common func end ---------------------

#ifdef __DAV_C220_CUBE__
template <typename DTYPE = float, typename MM_OUT_DTYPE = float>
class Conv1dAic {
public:
    __aicore__ __attribute__((always_inline)) inline Conv1dAic()
    {}

    __aicore__ __attribute__((always_inline)) inline void SetArgs(__gm__ uint8_t *__restrict__ sync,
        __gm__ uint8_t *__restrict__ inTensor, __gm__ uint8_t *__restrict__ kernelTensor,
        __gm__ uint8_t *__restrict__ outTensor, __gm__ uint8_t *__restrict__ workspace,
        __gm__ uint8_t *__restrict__ tilingGm)
    {
        SetFftsBaseAddr((uint64_t)sync);
        SetPadding<uint64_t>(0);
        SetAtomicnone();
        SetNdpara(1, 0, 0);
        SetMasknorm();

        signal_gm = reinterpret_cast<__gm__ DTYPE *>(inTensor);
        kernel_gm = reinterpret_cast<__gm__ DTYPE *>(kernelTensor);
        out_gm = reinterpret_cast<__gm__ DTYPE *>(outTensor);
        workspace_gm = reinterpret_cast<__gm__ DTYPE *>(workspace);

        signal_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE *>(signal_gm));
        kernel_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE *>(kernel_gm));
        out_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE *>(out_gm));
        workspace_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE *>(workspace_gm));

        sigLen = (int64_t)(*((__gm__ int64_t *)tilingGm));
        kernelLen = (int64_t)(*((__gm__ int64_t *)tilingGm + 1));
        batchCount = (int64_t)(*((__gm__ int64_t *)tilingGm + 2));

        kernelMatOriM = BASE_COL_BLOCK - kernelLen + 1;
        kernelMatOriN = 2;
        kernelMatOriK = BASE_COL_BLOCK;

        kernelMatComputeM = sigLen;
        kernelMatComputeN = 2;
        kernelMatComputeK = sigLen;
    }

    __aicore__ __attribute__((always_inline)) inline void Run()
    {
        uint32_t matOffset = kernelLen / 2;
        uint32_t totalBlockNum = AscendC::GetBlockNum();
        uint32_t currBlockId = AscendC::GetBlockIdx();

        // left mat size: BASE_COL_BLOCK x workspaceRowNum
        // right mat size: workspaceRowNum x BASE_COL_BLOCK

        int64_t workspaceColNum = BASE_COL_BLOCK + kernelLen * 2 - 2;
        int64_t workspaceRowNum = workspaceColNum + kernelLen * 2 - 2;
        int64_t computeNum = workspaceColNum;

        uint32_t currM = BASE_COL_BLOCK;
        uint32_t currK = computeNum;
        uint32_t currN = BASE_COL_BLOCK;
        uint64_t rowOffset = 0;
        uint64_t colOffset = 0;
        uint64_t startOffset = 0;

        uint32_t iterNum = sigLen * 2 / currN;
        uint32_t remainNum = sigLen * 2 % currN;

        uint32_t L1_PINGPONG_BUFFER_LEN = 256 * 1024 / sizeof(DTYPE);

        if (remainNum > 0)
            iterNum += 1;  // 3

        uint32_t currTaskId = -1;

        uint64_t signalBatchOffset = sigLen * 2;
        uint64_t outBatchOffset = sigLen * 2;

        uint32_t rowBlockIterNum = batchCount / BASE_COL_BLOCK;
        uint32_t rowRemainNum = batchCount % BASE_COL_BLOCK;
        if (rowRemainNum > 0)
            rowBlockIterNum++;

        WaitFlagDev(0x8);

        if (sigLen * 2 <= workspaceColNum) {
            for (uint32_t rowBlockIdx = 0; rowBlockIdx < rowBlockIterNum; rowBlockIdx++) {
                if (rowRemainNum > 0 && rowBlockIdx == rowBlockIterNum - 1)
                    currM = rowRemainNum;

                base_matmul_single_core_dev(signal_gm_tensor[rowBlockIdx * sigLen * 2 * BASE_COL_BLOCK],
                    workspace_gm_tensor[matOffset * workspaceColNum * 2],
                    out_gm_tensor[rowBlockIdx * sigLen * 2 * BASE_COL_BLOCK],
                    l1_base_a_tensor,
                    l1_base_b_tensor,
                    l0a_base_tensor,
                    l0b_base_tensor,
                    l0c_buf_tensor,
                    currM,
                    2 * sigLen,
                    2 * sigLen,  // M, N, K
                    batchCount,
                    2 * sigLen,
                    workspaceRowNum,
                    workspaceColNum,
                    batchCount,
                    2 * sigLen,
                    0,
                    0,
                    1,
                    1);
            }
            return;
        }

        uint32_t forwardIterNum = (sigLen * 2 - computeNum + BASE_COL_BLOCK - 1) / BASE_COL_BLOCK;
        uint32_t remainSigLen = sigLen * 2 - forwardIterNum * BASE_COL_BLOCK;
        uint32_t backwardIterNum = remainSigLen / BASE_COL_BLOCK;  // 1
        uint32_t backwardRemainSigLen = remainSigLen % BASE_COL_BLOCK;

        for (uint32_t taskId = currBlockId; taskId < forwardIterNum; taskId += totalBlockNum) {
            uint32_t leftMatOffset = taskId == 0 ? 0 : taskId * BASE_COL_BLOCK - 2 * matOffset;  // complex to float32
            uint32_t outOffset = taskId == 0 ? 0 : taskId * BASE_COL_BLOCK;
            uint32_t startOffset = taskId == 0 ? matOffset : 0;
            currM = BASE_COL_BLOCK;
            currK = computeNum - startOffset * 2;
            currN = BASE_COL_BLOCK;
            for (uint32_t rowBlockIdx = 0; rowBlockIdx < rowBlockIterNum; rowBlockIdx++) {
                if (rowRemainNum > 0 && rowBlockIdx == rowBlockIterNum - 1)
                    currM = rowRemainNum;

                base_matmul_single_core_dev(signal_gm_tensor[leftMatOffset + rowBlockIdx * sigLen * 2 * BASE_COL_BLOCK],
                    workspace_gm_tensor[startOffset * workspaceColNum * 2],
                    out_gm_tensor[outOffset + rowBlockIdx * sigLen * 2 * BASE_COL_BLOCK],
                    l1_base_a_tensor,
                    l1_base_b_tensor,
                    l0a_base_tensor,
                    l0b_base_tensor,
                    l0c_buf_tensor,
                    currM,
                    currN,
                    currK,  // M, N, K
                    batchCount,
                    2 * sigLen,
                    workspaceRowNum,
                    workspaceColNum,
                    batchCount,
                    2 * sigLen,
                    0,
                    0,
                    1,
                    1);
            }
        }

        if (remainSigLen > 0) {
            // compute remain signals
            currM = BASE_COL_BLOCK;
            for (uint32_t rowBlockIdx = currBlockId; rowBlockIdx < rowBlockIterNum; rowBlockIdx += totalBlockNum) {
                if (rowRemainNum > 0 && rowBlockIdx == rowBlockIterNum - 1)
                    currM = rowRemainNum;

                base_matmul_single_core_dev(signal_gm_tensor[forwardIterNum * BASE_COL_BLOCK - 2 * matOffset +
                                                             rowBlockIdx * sigLen * 2 * BASE_COL_BLOCK],
                    workspace_gm_tensor[0],
                    out_gm_tensor[forwardIterNum * BASE_COL_BLOCK + rowBlockIdx * sigLen * 2 * BASE_COL_BLOCK],
                    l1_base_a_tensor,
                    l1_base_b_tensor,
                    l0a_base_tensor,
                    l0b_base_tensor,
                    l0c_buf_tensor,
                    currM,
                    remainSigLen,
                    remainSigLen + 2 * matOffset,  // M, N, K
                    batchCount,
                    2 * sigLen,
                    workspaceRowNum,
                    workspaceColNum,
                    batchCount,
                    2 * sigLen,
                    0,
                    0,
                    1,
                    1);
            }
        }

        PIPE_BARRIER(ALL);
    }

private:
private:
    __gm__ DTYPE *__restrict__ signal_gm{nullptr};
    __gm__ DTYPE *__restrict__ kernel_gm{nullptr};
    __gm__ DTYPE *__restrict__ out_gm{nullptr};
    __gm__ DTYPE *__restrict__ workspace_gm{nullptr};

    AscendC::GlobalTensor<DTYPE> signal_gm_tensor;
    AscendC::GlobalTensor<DTYPE> kernel_gm_tensor;
    AscendC::GlobalTensor<DTYPE> out_gm_tensor;
    AscendC::GlobalTensor<DTYPE> workspace_gm_tensor;

    const uint32_t l1_base_a_offset = 0;
    const uint32_t l1_base_b_offset = 128 * 1024;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<DTYPE> l1_base_a_tensor = buf.GetBuffer<BufferType::ASCEND_CB, DTYPE>(0);
    AscendC::LocalTensor<DTYPE> l1_base_b_tensor = buf.GetBuffer<BufferType::ASCEND_CB, DTYPE>(l1_base_b_offset);

    AscendC::LocalTensor<DTYPE> l0a_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0A, DTYPE>(0);
    AscendC::LocalTensor<DTYPE> l0b_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0B, DTYPE>(0);
    AscendC::LocalTensor<MM_OUT_DTYPE> l0c_buf_tensor = buf.GetBuffer<BufferType::ASCEND_L0C, MM_OUT_DTYPE>(0);

    int64_t sigLen{0};
    int64_t kernelLen{0};
    int64_t batchCount{0};
    uint32_t kernelMatOriM{0};
    uint32_t kernelMatOriN{0};
    uint32_t kernelMatOriK{0};
    uint32_t kernelMatComputeM{0};
    uint32_t kernelMatComputeN{0};
    uint32_t kernelMatComputeK{0};
};
#endif

#ifdef __DAV_C220_VEC__
template <typename DTYPE = float, typename MM_OUT_DTYPE = float>
class Conv1dAiv {
public:
    __aicore__ __attribute__((always_inline)) inline Conv1dAiv()
    {}

    __aicore__ __attribute__((always_inline)) inline void SetArgs(__gm__ uint8_t *__restrict__ sync,
        __gm__ uint8_t *__restrict__ inTensor, __gm__ uint8_t *__restrict__ kernelTensor,
        __gm__ uint8_t *__restrict__ outTensor, __gm__ uint8_t *__restrict__ workspace,
        __gm__ uint8_t *__restrict__ tilingGm)
    {
        SetFftsBaseAddr((uint64_t)sync);
        sub_block_idx = static_cast<uint64_t>(GetSubBlockidx());
        SetAtomicnone();
        SetMasknorm();
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);

        signal_gm = reinterpret_cast<__gm__ DTYPE *>(inTensor);
        kernel_gm = reinterpret_cast<__gm__ DTYPE *>(kernelTensor);
        out_gm = reinterpret_cast<__gm__ DTYPE *>(outTensor);
        workspace_gm = reinterpret_cast<__gm__ DTYPE *>(workspace);

        signal_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE *>(signal_gm));
        kernel_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE *>(kernel_gm));
        out_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE *>(out_gm));
        workspace_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE *>(workspace_gm));

        sigLen = (int64_t)(*((__gm__ int64_t *)tilingGm));
        kernelLen = (int64_t)(*((__gm__ int64_t *)tilingGm + 1));
        batchCount = (int64_t)(*((__gm__ int64_t *)tilingGm + 2));
        dtype = (int64_t)(*((__gm__ int64_t *)tilingGm + 3));
    }

    __aicore__ __attribute__((always_inline)) inline void Run()
    {
        // construct kernel matrix
        uint32_t currBlockId = AscendC::GetBlockIdx();

        uint32_t workspaceColNum = BASE_COL_BLOCK + kernelLen * 2 - 2;
        uint32_t workspaceRowNum = workspaceColNum + kernelLen * 2 - 2;
        uint32_t baseBlockNum = 32 / sizeof(DTYPE);
        uint32_t totalNum = workspaceColNum * workspaceRowNum;
        uint32_t totalNumAligned = (workspaceColNum * workspaceRowNum + baseBlockNum - 1) / baseBlockNum * baseBlockNum;

        Duplicate(kernel_ubuf_tensor, (DTYPE)0, totalNumAligned);
        PIPE_BARRIER(ALL);
        for (int64_t i = 0; i < kernelLen; i++) {
            auto kernelVal = kernel_gm_tensor.GetValue(kernelLen - i - 1);
            for (int64_t rowIdx = i * 2; rowIdx < workspaceColNum + i * 2; rowIdx++) {
                kernel_ubuf_tensor.SetValue(rowIdx * workspaceColNum + rowIdx - i * 2, kernelVal);
            }
        }
        PIPE_BARRIER(ALL);

        AscendC::DataCopyExtParams param = {1, static_cast<uint32_t>(totalNum * sizeof(DTYPE)), 0, 0, 0};

        AscendC::DataCopyPad(workspace_gm_tensor, kernel_ubuf_tensor, param);
        PIPE_BARRIER(ALL);
        FftsCrossCoreSync<PIPE_MTE3, 0x2>(0x8);
    }

private:
    __gm__ DTYPE *__restrict__ signal_gm{nullptr};
    __gm__ DTYPE *__restrict__ kernel_gm{nullptr};
    __gm__ DTYPE *__restrict__ out_gm{nullptr};
    __gm__ DTYPE *__restrict__ workspace_gm{nullptr};

    AscendC::GlobalTensor<DTYPE> signal_gm_tensor;
    AscendC::GlobalTensor<DTYPE> kernel_gm_tensor;
    AscendC::GlobalTensor<DTYPE> out_gm_tensor;
    AscendC::GlobalTensor<DTYPE> workspace_gm_tensor;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<DTYPE> kernel_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, DTYPE>(0);

    uint64_t sub_block_idx{0};
    int64_t sigLen{0};
    int64_t kernelLen{0};
    int64_t batchCount{0};
    int64_t dtype{0};
};

#endif

extern "C" __global__ __aicore__ void convolve(__gm__ uint8_t *__restrict__ sync, __gm__ uint8_t *__restrict__ inTensor,
    __gm__ uint8_t *__restrict__ kernelTensor, __gm__ uint8_t *__restrict__ outTensor,
    __gm__ uint8_t *__restrict__ workspace, __gm__ uint8_t *__restrict__ tilingGm)
{
    SetFftsBaseAddr((unsigned long)sync);
    SetAtomicnone();
    SetMasknorm();
#ifdef __DAV_C220_VEC__
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
#elif __DAV_C220_CUBE__
    SetPadding<uint64_t>(0);
    SetNdpara(1, 0, 0);
#endif

    int64_t dtype = (int64_t)(*((__gm__ int64_t *)tilingGm + 3));
    if (dtype == 0) {
        // complex64
#ifdef __DAV_C220_CUBE__
        Conv1dAic<float, float> conv1d_aic{};
        conv1d_aic.SetArgs(sync, inTensor, kernelTensor, outTensor, workspace, tilingGm);
        conv1d_aic.Run();
#endif
#ifdef __DAV_C220_VEC__
        Conv1dAiv<float, float> conv1d_aiv{};
        conv1d_aiv.SetArgs(sync, inTensor, kernelTensor, outTensor, workspace, tilingGm);
        conv1d_aiv.Run();
#endif
    } else if (dtype == 1) {
        // complex32
#ifdef __DAV_C220_CUBE__
        Conv1dAic<half, float> conv1d_aic{};
        conv1d_aic.SetArgs(sync, inTensor, kernelTensor, outTensor, workspace, tilingGm);
        conv1d_aic.Run();
#endif
#ifdef __DAV_C220_VEC__
        Conv1dAiv<half, float> conv1d_aiv{};
        conv1d_aiv.SetArgs(sync, inTensor, kernelTensor, outTensor, workspace, tilingGm);
        conv1d_aiv.Run();
#endif
    }
}