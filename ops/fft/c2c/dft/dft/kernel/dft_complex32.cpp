/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 2201
#endif

#include "kernel_operator.h"
#include "dft_mix_kernel.h"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

using namespace Catlass;

constexpr uint32_t L1_SHAPE_M = 128;
constexpr uint32_t L1_SHAPE_N = 128;
constexpr uint32_t L1_SHAPE_K = 256;

constexpr uint32_t L0_SHAPE_M = 128;
constexpr uint32_t L0_SHAPE_N = 128;
constexpr uint32_t L0_SHAPE_K = 64;

extern "C" __global__ __aicore__ void dft_complex32(
    GM_ADDR ffts_addr,
    GM_ADDR gm_a,
    GM_ADDR gm_b,
    GM_ADDR gm_c,
    GM_ADDR workspace,
    GM_ADDR tiling_para_gm
) {
    AscendC::SetSyncBaseAddr((uint64_t)ffts_addr);
    uint32_t batchSize, m, n, k, trans_b;

    auto tiling_para = reinterpret_cast<__gm__ int32_t *>(tiling_para_gm);
    batchSize = tiling_para[0]; // set to 1
    m = tiling_para[1]; // batch size
    n = tiling_para[2]; // fftN * 2
    k = tiling_para[3]; // fftN * 2
    trans_b = tiling_para[5];
    int32_t loopTime = tiling_para[9];
    int32_t batchNumPreLoop = tiling_para[10];
    int32_t batchNumPreCore = tiling_para[11];
    int32_t batchTailNum = tiling_para[12];
    int32_t batchTailNumPreCore = tiling_para[13];
    int32_t batchTailCoreNum = tiling_para[14];

    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
    using L1TileShape = GemmShape<L1_SHAPE_M, L1_SHAPE_N, L1_SHAPE_K>;
    using L0TileShape = GemmShape<L0_SHAPE_M, L0_SHAPE_N, L0_SHAPE_K>;

    using AType = Gemm::GemmType<float, layout::RowMajor>;
    using CType = Gemm::GemmType<float, layout::RowMajor>;
    using BlockEpilogue = void;
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    
    Catlass::GemmCoord problemShape{m, n, k};
    Arch::Resource<ArchTag> resource;
    if (trans_b) {
        using BType = Gemm::GemmType<float, layout::ColumnMajor>;
        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        using DftMixKernel = Gemm::Kernel::DftMixKernel<half, BlockMmad, BlockEpilogue, BlockScheduler>;
        DftMixKernel::Params params{problemShape, gm_a, gm_b, gm_c, workspace, loopTime, batchNumPreLoop,
                                    batchNumPreCore, batchTailNum, batchTailNumPreCore, batchTailCoreNum};
        DftMixKernel kernel_op;
        kernel_op(params, resource);
    } else {
        using BType = Gemm::GemmType<float, layout::RowMajor>;
        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        using DftMixKernel = Gemm::Kernel::DftMixKernel<half, BlockMmad, BlockEpilogue, BlockScheduler>;
        DftMixKernel::Params params{problemShape, gm_a, gm_b, gm_c, workspace, loopTime, batchNumPreLoop,
                                    batchNumPreCore, batchTailNum, batchTailNumPreCore, batchTailCoreNum};
        DftMixKernel kernel_op;
        kernel_op(params, resource);
    }
}
