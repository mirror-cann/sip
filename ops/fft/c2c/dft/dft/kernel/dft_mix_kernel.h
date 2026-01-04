/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DFT_MIX_KERNEL_H
#define DFT_MIX_KERNEL_H

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Gemm::Kernel {

constexpr int32_t SYNC_MODE_AIC_AIV = 0;    // inter block synchronization
constexpr int32_t SYNC_MODE_SUB_BLOCK = 1;  // inter subblock synchronization
constexpr int32_t SYNC_MODE_GROUP = 2;      // intra block synchronization

constexpr int32_t AICFLAGID = 0;
constexpr int32_t AIVFLAGID = 1;
constexpr int32_t AIC2AIVFLAGID = 2;
constexpr int32_t AIV2AICFLAGID = 3;
constexpr int32_t AIC_FINISH_FLAG_ID = 6;
constexpr int32_t AIV_FINISH_FLAG_ID = 7;

constexpr int64_t UB_SIZE = 192 * 1024;
constexpr int64_t CUBE_CORE_NUM = 20;
constexpr int64_t VEC_CORE_NUM = 40;
constexpr int64_t MAX_WORKSPACE_OFFSET = UB_SIZE * VEC_CORE_NUM / 3 * 2 / sizeof(float);

template <
    class  DType,
    class  BlockMmad_,
    class  BlockEpilogue_,
    class  BlockScheduler_
>
class DftMixKernel {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
        GM_ADDR ptrWorkspace;
        int32_t loopTime;
        int32_t batchNumPreLoop;
        int32_t batchNumPreCore;
        int32_t batchTailNum;
        int32_t batchTailNumPreCore;
        int32_t batchTailCoreNum;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GM_ADDR ptrA_, GM_ADDR ptrB_,
               GM_ADDR ptrC_, GM_ADDR ptrWorkspace_, int32_t loopTime_,
               int32_t batchNumPreLoop_, int32_t batchNumPreCore_, int32_t batchTailNum_,
               int32_t batchTailNumPreCore_, int32_t batchTailCoreNum_)
            : problemShape(problemShape_), ptrA(ptrA_), ptrB(ptrB_),
              ptrC(ptrC_), ptrWorkspace(ptrWorkspace_), loopTime(loopTime_),
              batchNumPreLoop(batchNumPreLoop_), batchNumPreCore(batchNumPreCore_), batchTailNum(batchTailNum_),
              batchTailNumPreCore(batchTailNumPreCore_), batchTailCoreNum(batchTailCoreNum_) {}
    };

    // Methods
    CATLASS_DEVICE
    DftMixKernel() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params, Arch::Resource<ArchTag> &resource);

    // Executes one Matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params, Arch::Resource<ArchTag> &resource) {
        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrWorkspace);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace);

        int64_t pingPongFlag = 0;
        for (int i = 0; i < params.loopTime; i++) {
            uint32_t curBatchNum = (i == params.loopTime - 1) ? params.batchTailNum : params.batchNumPreLoop;
            Catlass::GemmCoord problemShape{curBatchNum, params.problemShape.n(), params.problemShape.k()};
            LayoutA layoutA{curBatchNum, params.problemShape.k()};
            LayoutB layoutB{params.problemShape.k(), params.problemShape.n()};
            LayoutC layoutC{curBatchNum, params.problemShape.n()};

            BlockScheduler matmulBlockScheduler(problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
            BlockMmad blockMmad(resource);

            AscendC::CrossCoreWaitFlag(AIV_FINISH_FLAG_ID);
            for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                // Compute block location
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                // Compute initial location in logical coordinates
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);

                // Compute block-scoped matrix multiply-add
                blockMmad(gmA[gmOffsetA + pingPongFlag * MAX_WORKSPACE_OFFSET], layoutA,
                          gmB[gmOffsetB], layoutB,
                          gmC[gmOffsetC + pingPongFlag * MAX_WORKSPACE_OFFSET + MAX_WORKSPACE_OFFSET * 2], layoutC,
                          actualBlockShape);
                AscendC::PipeBarrier<PIPE_ALL>();
            }

            pingPongFlag = 1- pingPongFlag;
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::CrossCoreSetFlag<SYNC_MODE_AIC_AIV, PIPE_FIX>(AIC_FINISH_FLAG_ID);
            AscendC::CrossCoreWaitFlag(AIC_FINISH_FLAG_ID);
            AscendC::CrossCoreSetFlag<SYNC_MODE_GROUP, PIPE_FIX>(AIC_FINISH_FLAG_ID);
        }
    }

    template <class IN_TYPE, class OUT_TYPE>
    CATLASS_DEVICE
    void innerCast(
        uint32_t length,
        AscendC::LocalTensor<IN_TYPE> &inUb,
        AscendC::LocalTensor<OUT_TYPE> &outUb,
        const AscendC::GlobalTensor<IN_TYPE> &gmIn,
        const AscendC::GlobalTensor<OUT_TYPE> &gmOut)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::DataCopyPadExtParams<IN_TYPE> padExtParams{false, 0, 0, 0};
        AscendC::DataCopyPad(inUb, gmIn, {1, static_cast<uint32_t>(length * sizeof(IN_TYPE)), 0, 0, 0}, padExtParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(outUb, inUb, AscendC::RoundMode::CAST_NONE, length);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(gmOut, outUb, {1, static_cast<uint32_t>(length * sizeof(OUT_TYPE)), 0, 0, 0});
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params, Arch::Resource<ArchTag> &resource) {
        int64_t blockIdx = AscendC::GetBlockIdx();
        int64_t subBlockIdx = AscendC::GetSubBlockIdx();
        int64_t subBlockNum = AscendC::GetSubBlockNum();
        int64_t blockNum = AscendC::GetBlockNum();
        int64_t baseBlockIdx = blockIdx / subBlockNum;
        
        AscendC::GlobalTensor<DType> gmA;
        gmA.SetGlobalBuffer((__gm__ DType *)params.ptrA);
        AscendC::GlobalTensor<float> gmWorkspace;
        gmWorkspace.SetGlobalBuffer((__gm__ float *)params.ptrWorkspace);
        AscendC::GlobalTensor<DType> gmC;
        gmC.SetGlobalBuffer((__gm__ DType *)params.ptrC);

        uint32_t maxlength = ArchTag::UB_SIZE / 3;
        AscendC::LocalTensor<DType> inUb = resource.ubBuf.template GetBufferByByte<DType>(maxlength * 2);
        AscendC::LocalTensor<float> castUB = resource.ubBuf.template GetBufferByByte<float>(0);

        int64_t pingPongFlag = 0;
        int64_t tmpOffset = 0;
        int64_t tmpWorkspaceOffset = 0;
        uint32_t tmpLength = 0;
        for (int i = 0; i < params.loopTime; i++) {
            uint32_t length = 0;
            int64_t innerOffset = 0;
            int64_t offset = i * params.batchNumPreLoop * params.problemShape.n();
            int64_t workspaceOffset = pingPongFlag * MAX_WORKSPACE_OFFSET;
            if (i == params.loopTime - 1) {
                if (blockIdx < params.batchTailCoreNum) {
                    length = (params.batchTailNumPreCore + 1) * params.problemShape.n();
                    innerOffset = blockIdx * length;
                } else {
                    length = params.batchTailNumPreCore * params.problemShape.n();
                    innerOffset = blockIdx * length + params.batchTailCoreNum * params.problemShape.n();
                }
            } else {
                length = params.batchNumPreCore * params.problemShape.n();
                innerOffset = blockIdx * length;
            }
            offset += innerOffset;
            workspaceOffset += innerOffset;

            innerCast(length, inUb, castUB, gmA[offset], gmWorkspace[workspaceOffset]);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::CrossCoreSetFlag<SYNC_MODE_AIC_AIV, PIPE_MTE3>(AIV_FINISH_FLAG_ID);
            AscendC::CrossCoreWaitFlag(AIV_FINISH_FLAG_ID);
            AscendC::CrossCoreSetFlag<SYNC_MODE_GROUP, PIPE_MTE3>(AIV_FINISH_FLAG_ID);

            pingPongFlag = 1 - pingPongFlag;
            if (i == 0) {
                tmpLength = length;
                tmpOffset = offset;
                tmpWorkspaceOffset = workspaceOffset;
                continue;
            }
            AscendC::CrossCoreWaitFlag(AIC_FINISH_FLAG_ID);
            innerCast(tmpLength, castUB, inUb, gmWorkspace[tmpWorkspaceOffset + MAX_WORKSPACE_OFFSET * 2], gmC[tmpOffset]);
            tmpLength = length;
            tmpOffset = offset;
            tmpWorkspaceOffset = workspaceOffset;
        }
        AscendC::CrossCoreWaitFlag(AIC_FINISH_FLAG_ID);
        innerCast(tmpLength, castUB, inUb, gmWorkspace[tmpWorkspaceOffset + MAX_WORKSPACE_OFFSET * 2], gmC[tmpOffset]);
    }
};

} // namespace Catlass::Gemm::Kernel

#endif // DFT_MIX_KERNEL_H