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
#include "kernel/fft_batched_matmul_kernel.hpp"
#include "block/block_epilogue_complex_mul.hpp"
#include "block/block_epilogue_transpose.hpp"
#include "tile/tile_complex_mul.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

using T_INPUT = float;
using T_OUTPUT = float;

using namespace Catlass;

constexpr uint32_t L1_SHAPE_M = 128;
constexpr uint32_t L1_SHAPE_N = 128;
constexpr uint32_t L1_SHAPE_K = 128;

constexpr uint32_t L0_SHAPE_M = 128;
constexpr uint32_t L0_SHAPE_N = 128;
constexpr uint32_t L0_SHAPE_K = 64;

constexpr uint32_t MATMUL_READY_ID = 1;
constexpr uint32_t VEC_COMPUTE_READY_ID = 2;

constexpr uint32_t MAX_ITERATION = 10;
constexpr uint32_t STAGES = 2;

struct FFTBSepKernelParams {
    GM_ADDR gm_input_real;
    GM_ADDR gm_input_imag;
    GM_ADDR gm_mat_w;
    GM_ADDR gm_mat_t;
    GM_ADDR gm_out_real;
    GM_ADDR gm_out_imag;
    GM_ADDR workspace;
    GM_ADDR tiling_para_gm;
    uint32_t fftLen;
    uint32_t batchSize;
    uint32_t direction;
    uint32_t iterCount;
    uint32_t radixVec[MAX_ITERATION];

    // Methods
    CATLASS_DEVICE
    FFTBSepKernelParams() {
    }

    CATLASS_DEVICE
    FFTBSepKernelParams(
        GM_ADDR gm_input_real_,
        GM_ADDR gm_input_imag_,
        GM_ADDR gm_mat_w_,
        GM_ADDR gm_mat_t_,
        GM_ADDR gm_out_real_,
        GM_ADDR gm_out_imag_,
        GM_ADDR workspace_,
        GM_ADDR tiling_para_gm_,
        uint32_t fftLen_,
        uint32_t batchSize_,
        uint32_t direction_,
        uint32_t iterCount_,
        uint32_t *radixVec_)
        : gm_input_real(gm_input_real_),
          gm_input_imag(gm_input_imag_),
          gm_mat_w(gm_mat_w_),
          gm_mat_t(gm_mat_t_),
          gm_out_real(gm_out_real_),
          gm_out_imag(gm_out_imag_),
          workspace(workspace_),
          tiling_para_gm(tiling_para_gm_),
          fftLen(fftLen_),
          batchSize(batchSize_),
          direction(direction_),
          iterCount(iterCount_) {
            for (uint32_t i = 0; i < MAX_ITERATION; i++) {
                radixVec[i] = radixVec_[i];
            }
    }
};


template<class MatmulKernelForDft, class BlockEpilogueComplexMul_, class BlockEpilogueTranspose_>
class FFTBSepKernel {
public:
    using MatmulKernel = MatmulKernelForDft;
    using ArchTag = typename MatmulKernelForDft::ArchTag;
    using MatmulParams = typename MatmulKernelForDft::Params;
    using ElementA = typename MatmulKernelForDft::ElementA;
    using LayoutA = typename MatmulKernelForDft::LayoutA;
    using ElementB = typename MatmulKernelForDft::ElementB;
    using LayoutB = typename MatmulKernelForDft::LayoutB;
    using ElementC = typename MatmulKernelForDft::ElementC;
    using LayoutC = typename MatmulKernelForDft::LayoutC;

    using BlockEpilogueComplexMul = BlockEpilogueComplexMul_;
    using BlockEpilogueTranspose = BlockEpilogueTranspose_;

    using ComplexMulParams = typename BlockEpilogueComplexMul::Params;
    using TransposeParams = typename BlockEpilogueTranspose::Params;
    using ScaleLayout = typename BlockEpilogueComplexMul::ScaleLayout;

    // Methods
    CATLASS_DEVICE
    FFTBSepKernel() {
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(FFTBSepKernelParams const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(FFTBSepKernelParams const &params) {
        auto radixVec = params.radixVec;
        uint32_t fftLen = params.fftLen;
        uint32_t batchSize = params.batchSize;
        uint32_t iterCount = params.iterCount;
        uint32_t direction = params.direction;
        
        auto inputReal = params.gm_input_real;
        auto inputImag = params.gm_input_imag;
        auto outputReal = params.gm_out_real;
        auto outputImag = params.gm_out_imag;
        auto workspaceReal1 = params.workspace;
        auto workspaceImag1 = params.workspace + fftLen * batchSize * sizeof(T_OUTPUT);
        auto workspaceReal2 = params.workspace + fftLen * batchSize * sizeof(T_OUTPUT) * 2;
        auto workspaceImag2 = params.workspace + fftLen * batchSize * sizeof(T_OUTPUT) * 3;
        auto wMatrix = params.gm_mat_w;
        auto tMatrix = params.gm_mat_t;
        uint64_t wOffset = 0;
        uint64_t tOffset = 0;
        int64_t strideA = 0;
        int64_t strideB = 0;
        int64_t strideC = 0;

        MatmulKernel matmulOp;
        Catlass::GemmCoord problemShape{128, 128, 128};

        MatmulParams matmulParams;

        uint32_t innerBatch = 1;
        uint32_t remain = fftLen;
        uint32_t currBatch = 1;

        for (uint32_t idx = 0; idx < iterCount; idx++) {
            auto currInGmReal = idx == 0 ? inputReal : workspaceReal1;
            auto currInGmImag = idx == 0 ? inputImag : workspaceImag1;
            auto currOutGmReal = workspaceReal2;
            auto currOutGmImag = workspaceImag2;

            uint32_t radix = radixVec[idx];
            remain = remain / radix;

            currBatch = batchSize * innerBatch;
            problemShape.m() = radix;
            problemShape.k() = radix;
            problemShape.n() = remain;

            auto wMatrixReal = wMatrix + wOffset;
            auto wMatrixImag = wMatrix + wOffset + radix * radix * sizeof(float);
            auto wMatrixNegImag = wMatrix + wOffset + radix * radix * sizeof(float) * 2;

            wOffset += radix * radix * sizeof(float) * 3;

            Arch::CrossCoreWaitFlag(vecComputeReady);

            AscendC::SetAtomicNone();

            matmulParams.batchCount = currBatch;
            matmulParams.problemShape = problemShape;
            matmulParams.layoutA = {radix, radix};
            matmulParams.strideA = 0;
            matmulParams.layoutB = {radix, remain};
            matmulParams.strideB = radix * remain;
            matmulParams.layoutC = {radix, remain};
            matmulParams.strideC = radix * remain;

            // R * R
            matmulParams.ptrA = wMatrixReal;
            matmulParams.ptrB = currInGmReal;
            matmulParams.ptrC = currOutGmReal;
            matmulOp(matmulParams, resource);

            // R * I
            matmulParams.ptrA = wMatrixReal;
            matmulParams.ptrB = currInGmImag;
            matmulParams.ptrC = currOutGmImag;
            matmulOp(matmulParams, resource);

            AscendC::SetAtomicAdd<float>();

            // I * I
            matmulParams.ptrA = wMatrixNegImag;
            matmulParams.ptrB = currInGmImag;
            matmulParams.ptrC = currOutGmReal;
            matmulOp(matmulParams, resource);

            // I * R
            matmulParams.ptrA = wMatrixImag;
            matmulParams.ptrB = currInGmReal;
            matmulParams.ptrC = currOutGmImag;
            matmulOp(matmulParams, resource);

            AscendC::SetAtomicNone();

            Arch::CrossCoreSetFlag<0x0, PIPE_FIX>(matmulReady);
            Arch::CrossCoreWaitFlag(matmulReady);
            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(matmulReady);

            innerBatch = innerBatch * radix;
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(FFTBSepKernelParams const &params) {
        auto radixVec = params.radixVec;
        uint32_t fftLen = params.fftLen;
        uint32_t batchSize = params.batchSize;
        uint32_t iterCount = params.iterCount;
        uint32_t direction = params.direction;
        
        auto inputReal = params.gm_input_real;
        auto inputImag = params.gm_input_imag;
        auto outputReal = params.gm_out_real;
        auto outputImag = params.gm_out_imag;
        auto wMatrix = params.gm_mat_w;
        auto tMatrix = params.gm_mat_t;
        auto workspaceReal1 = params.workspace;
        auto workspaceImag1 = params.workspace + fftLen * batchSize * sizeof(T_OUTPUT);
        auto workspaceReal2 = params.workspace + fftLen * batchSize * sizeof(T_OUTPUT) * 2;
        auto workspaceImag2 = params.workspace + fftLen * batchSize * sizeof(T_OUTPUT) * 3;

        uint64_t tOffset = 0;

        uint32_t innerBatch = 1;
        uint32_t remain = fftLen;
        uint32_t currBatch = 1;
        Arch::CrossCoreSetFlag<0x0, PIPE_MTE3>(vecComputeReady);
        Arch::CrossCoreWaitFlag(vecComputeReady);
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecComputeReady);

        ComplexMulParams complexParams;
        TransposeParams transParams;

        BlockEpilogueComplexMul complexMulOp(resource);
        BlockEpilogueTranspose transposeOp(resource);

        for (uint32_t idx = 0; idx < iterCount; idx++) {
            auto currInGmReal = workspaceReal2;
            auto currInGmImag = workspaceImag2;

            auto currMulOutGmReal = idx == iterCount - 1 ? workspaceReal1 : outputReal;
            auto currMulOutGmImag = idx == iterCount - 1 ? workspaceImag1 : outputImag;

            auto currTransOutGmReal = idx == iterCount - 1 ? outputReal : workspaceReal1;
            auto currTransOutGmImag = idx == iterCount - 1 ? outputImag : workspaceImag1;

            // complex matmul and transpose
            uint32_t radix = radixVec[idx];
            remain = remain / radix;

            currBatch = batchSize * innerBatch;

            auto tMatrixReal = tMatrix + tOffset;
            auto tMatrixImag = tMatrix + tOffset + radix * remain * sizeof(float);

            // x shape: [currBatch, radix, remain]
            // t matrix shape: [radix, remain]
            Arch::CrossCoreWaitFlag(matmulReady);

            for (uint32_t vecCoreIdx = AscendC::GetBlockIdx(); vecCoreIdx < currBatch; vecCoreIdx += AscendC::GetBlockNum()) {
                uint32_t batchOffsetByte = radix * remain * sizeof(T_OUTPUT) * vecCoreIdx;
                complexParams = {
                    currInGmReal + batchOffsetByte,
                    currInGmImag + batchOffsetByte,
                    tMatrixReal, tMatrixImag,
                    currMulOutGmReal + batchOffsetByte,
                    currMulOutGmImag + batchOffsetByte,
                    1, {radix * remain},
                    radix * remain, 0, radix * remain};
                
                complexMulOp(complexParams);
            }

            Arch::CrossCoreSetFlag<0x0, PIPE_MTE3>(vecComputeReady);
            Arch::CrossCoreWaitFlag(vecComputeReady);

            for (uint32_t vecCoreIdx = AscendC::GetBlockIdx(); vecCoreIdx < batchSize; vecCoreIdx += AscendC::GetBlockNum()) {
                uint32_t batchOffsetByte = fftLen * sizeof(T_OUTPUT) * vecCoreIdx;
                transParams = {
                    currMulOutGmReal + batchOffsetByte,
                    currMulOutGmImag + batchOffsetByte,
                    currTransOutGmReal + batchOffsetByte,
                    currTransOutGmImag + batchOffsetByte,
                    {innerBatch, radix, remain}};
                
                transposeOp(transParams);
            }

            innerBatch = innerBatch * radix;

            Arch::CrossCoreSetFlag<0x0, PIPE_MTE3>(vecComputeReady);
            Arch::CrossCoreWaitFlag(vecComputeReady);
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecComputeReady);

            tOffset += radix * remain * sizeof(float) * 2;
        }
    }

private:
    Arch::Resource<ArchTag> resource;
    Arch::CrossCoreFlag matmulReady{MATMUL_READY_ID};
    Arch::CrossCoreFlag vecComputeReady{VEC_COMPUTE_READY_ID};
};


extern "C" CATLASS_GLOBAL void fft_b_sep(
    GM_ADDR ffts_addr,
    GM_ADDR gm_input_real,
    GM_ADDR gm_input_imag,
    GM_ADDR gm_mat_w,
    GM_ADDR gm_mat_t,
    GM_ADDR gm_out_real,
    GM_ADDR gm_out_imag,
    GM_ADDR workspace,
    GM_ADDR tiling_para_gm
) {
    AscendC::SetSyncBaseAddr((uint64_t)ffts_addr);

    auto tiling_para = reinterpret_cast<__gm__ uint32_t *>(tiling_para_gm);
    uint32_t fftLen = tiling_para[0]; // signal length
    uint32_t batchSize = tiling_para[1]; // batch size
    uint32_t direction = tiling_para[2]; // dicrection
    uint32_t iterCount = tiling_para[3]; // num of radix
    uint32_t radixVec[MAX_ITERATION]; // radix list

    for (uint32_t i = 0; i < MAX_ITERATION; i++) {
        radixVec[i] = tiling_para[i + 4];
    }

    // matmul init
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    layout::VectorLayout layoutScale{batchSize * fftLen};
    using AType = Gemm::GemmType<T_INPUT, LayoutA>;
    using BType = Gemm::GemmType<T_INPUT, LayoutB>;
    using CType = Gemm::GemmType<T_OUTPUT, LayoutC>;
    using ScaleType = Gemm::GemmType<T_OUTPUT, layout::VectorLayout>;
    using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
    using L1TileShape = GemmShape<L1_SHAPE_M, L1_SHAPE_N, L1_SHAPE_K>;
    using L0TileShape = GemmShape<L0_SHAPE_M, L0_SHAPE_N, L0_SHAPE_K>;
    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;
    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    // kernel level
    using MatmulKernel = Gemm::Kernel::FftBatchedMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;


    // vector init
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileElemWiseComplexMul = Epilogue::Tile::TileComplexMul<ArchTag, ScaleType, 1024>;
    using BlockEpilogueComplexMul = Epilogue::Block::BlockEpilogueComplexMul<ScaleType, TileElemWiseComplexMul, STAGES>;
    using BlockEpilogueTranspose = Epilogue::Block::BlockEpilogueTranspose<ArchTag, CType, STAGES>;


    using FFTBSepKernel = FFTBSepKernel<MatmulKernel, BlockEpilogueComplexMul, BlockEpilogueTranspose>;

    FFTBSepKernel fftBSepOp;

    FFTBSepKernelParams params{gm_input_real, gm_input_imag, gm_mat_w, gm_mat_t, gm_out_real, gm_out_imag, workspace, tiling_para_gm,
        fftLen, batchSize, direction, iterCount, radixVec};

    fftBSepOp(params);
}
