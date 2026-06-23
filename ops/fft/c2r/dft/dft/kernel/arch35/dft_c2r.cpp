/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#define CATLASS_ARCH 3510

#include "kernel_operator.h"

#include "catlass/gemm/kernel/basic_matmul_tla.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

using namespace Catlass;
using namespace tla;

extern "C" __global__ __aicore__ void dft_c2r(
    GM_ADDR gm_a,
    GM_ADDR gm_b,
    GM_ADDR gm_c,
    __gm__ uint8_t * __restrict__ workspace,
    __gm__ uint8_t * __restrict__ tiling_para_gm)
{
    AscendC::GlobalTensor<uint64_t> global;
    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(global);

    int32_t batchSize, trans_a, trans_b;
    uint32_t m, n, k;
    
    // get tiling args
    auto tiling_para = reinterpret_cast<__gm__ int32_t *>(tiling_para_gm);
    batchSize = tiling_para[0];
    m = static_cast<uint32_t>(tiling_para[1]);
    n = static_cast<uint32_t>(tiling_para[2]);
    k = static_cast<uint32_t>(tiling_para[3]);
    trans_a = tiling_para[4];
    trans_b = tiling_para[5];

    // set matmul problem shape
    GemmCoord problemShape{static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k)};

    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    // if no bias, set ElementBias to void
    using ElementBias = void;

    using ElementBiasType = std::conditional_t<std::is_void_v<ElementBias>, uint8_t, ElementBias>;

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::RowMajor;
    using LayoutTagC = layout::RowMajor;
    LayoutTagA tagA = LayoutTagA::MakeLayout<ElementA>(m, k);
    LayoutTagB tagB = LayoutTagB::MakeLayout<ElementB>(k, n);
    LayoutTagC tagC = LayoutTagC::MakeLayout<ElementC>(m, n);

    size_t lenA = tagA.Capacity();
    size_t lenB = tagB.Capacity();
    size_t lenC = tagC.Capacity();
    size_t lenBias = static_cast<size_t>(n);

    size_t sizeA = lenA * sizeof(ElementA);
    size_t sizeB = lenB * sizeof(ElementB);
    size_t sizeC = lenC * sizeof(ElementC);
    size_t sizeBias = lenBias * sizeof(ElementBiasType);

    using ArchTag = Arch::Ascend950;
    constexpr bool enableUnitFlag = true;
    constexpr bool useHF32 = false;
    constexpr bool enableL1Resident = false;
    constexpr uint32_t l0CStages = 1;
    constexpr uint32_t l1AStages = 2;
    constexpr uint32_t l1BStages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    using DispatchPolicy = Gemm::MmadPingpong<
        ArchTag,
        enableUnitFlag, useHF32, l0CStages, enableL1Resident,
        l1AStages, l1BStages, l0AStages, l0BStages
    >;

    using L1TileShape = Shape<Int<256>, Int<256>, Int<128>>;
    using L0TileShape = Shape<Int<256>, Int<256>, Int<32>>;
    
    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);

    using TileCopy = Gemm::Tile::PackedTileCopyTla<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, ElementBias>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy>;
    using BlockEpilogue = void;

    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    using MatmulKernel = Gemm::Kernel::BasicMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler>;

    MatmulKernel::Params params{
        problemShape, gm_a, layoutA, gm_b, layoutB, gm_c, layoutC, nullptr
    };

    MatmulKernel matmulOp;

    // run the matmul kernel.
    matmulOp(params);

}
