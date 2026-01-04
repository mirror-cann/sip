 /**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_EPICLOGUE_COMPLEX_MUL_HPP
#define BLOCK_EPICLOGUE_COMPLEX_MUL_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemv/tile/vec_copy_gm_to_ub.hpp"
#include "catlass/gemv/tile/vec_copy_ub_to_gm.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/gemv_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Epilogue::Block {

template <
    class DataType_,
    class TileElemWiseEpilogueComplexMul_,
    uint32_t STAGES_
>
class BlockEpilogueComplexMul {
public:
    using TileElemWiseEpilogueComplexMul = TileElemWiseEpilogueComplexMul_;
    using Element = typename DataType_::Element;
    using ArchTag = typename TileElemWiseEpilogueComplexMul::ArchTag;
    using ScaleLayout = typename DataType_::Layout;
    using VecCopyGmToUb = Catlass::Gemv::Tile::VecCopyGmToUB<ArchTag, DataType_>;
    using VecCopyUbToGm = Gemv::Tile::VecCopyUBToGm<ArchTag, DataType_>;

    static constexpr uint32_t COMPUTE_LENGTH = TileElemWiseEpilogueComplexMul::COMPUTE_LENGTH;
    static constexpr uint32_t STAGES = STAGES_;

    struct Params {
        GM_ADDR ptrSrc1Real;
        GM_ADDR ptrSrc1Imag;
        GM_ADDR ptrSrc2Real;
        GM_ADDR ptrSrc2Imag;
        GM_ADDR ptrOutReal;
        GM_ADDR ptrOutImag;

        uint32_t batchSize;
        ScaleLayout layout;

        int64_t src1Stride;
        int64_t src2Stride;
        int64_t outStride;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GM_ADDR ptrSrc1Real_,
            GM_ADDR ptrSrc1Imag_,
            GM_ADDR ptrSrc2Real_,
            GM_ADDR ptrSrc2Imag_,
            GM_ADDR ptrOutReal_,
            GM_ADDR ptrOutImag_,

            uint32_t batchSize_,
            ScaleLayout layout_,
            int64_t src1Stride_,
            int64_t src2Stride_,
            int64_t outStride_)
            : ptrSrc1Real(ptrSrc1Real_), ptrSrc1Imag(ptrSrc1Imag_),
              ptrSrc2Real(ptrSrc2Real_), ptrSrc2Imag(ptrSrc2Imag_),
              ptrOutReal(ptrOutReal_), ptrOutImag(ptrOutImag_),
              batchSize(batchSize_), layout(layout_),
              src1Stride(src1Stride_), src2Stride(src2Stride_),
              outStride(outStride_) {}
    };

    // construct
    CATLASS_DEVICE
    BlockEpilogueComplexMul(Arch::Resource<ArchTag>& resource, uint32_t ubByteStart = 0)
    {
        uint32_t numTensorList = 7;
        uint32_t tileSizeByByte = ArchTag::UB_SIZE / numTensorList / STAGES;
        tileSizeByByte = tileSizeByByte / 32 * 32;

        for (uint32_t i = 0; i < STAGES; ++i) {
            ubSrc1RealTensor[i] = resource.ubBuf.template GetBufferByByte<Element>(ubByteStart);
            ubByteStart += tileSizeByByte;
            ubSrc1ImagTensor[i] = resource.ubBuf.template GetBufferByByte<Element>(ubByteStart);
            ubByteStart += tileSizeByByte;
            ubSrc2RealTensor[i] = resource.ubBuf.template GetBufferByByte<Element>(ubByteStart);
            ubByteStart += tileSizeByByte;
            ubSrc2ImagTensor[i] = resource.ubBuf.template GetBufferByByte<Element>(ubByteStart);
            ubByteStart += tileSizeByByte;
            ubOutRealTensor[i] = resource.ubBuf.template GetBufferByByte<Element>(ubByteStart);
            ubByteStart += tileSizeByByte;
            ubOutImagTensor[i] = resource.ubBuf.template GetBufferByByte<Element>(ubByteStart);
            ubByteStart += tileSizeByByte;
            ubTempTensor[i] = resource.ubBuf.template GetBufferByByte<Element>(ubByteStart);
            ubByteStart += tileSizeByByte;

            UbEventList[i] = i;
        }
    }

    CATLASS_DEVICE
    ~BlockEpilogueComplexMul()
    {}

    CATLASS_DEVICE
    void operator()(Params const& params)
    {
        AscendC::GlobalTensor<Element> src1RealGm;
        src1RealGm.SetGlobalBuffer((__gm__ Element *)params.ptrSrc1Real);
        AscendC::GlobalTensor<Element> src1ImagGm;
        src1ImagGm.SetGlobalBuffer((__gm__ Element *)params.ptrSrc1Imag);
        AscendC::GlobalTensor<Element> src2RealGm;
        src2RealGm.SetGlobalBuffer((__gm__ Element *)params.ptrSrc2Real);
        AscendC::GlobalTensor<Element> src2ImagGm;
        src2ImagGm.SetGlobalBuffer((__gm__ Element *)params.ptrSrc2Imag);
        AscendC::GlobalTensor<Element> outRealGm;
        outRealGm.SetGlobalBuffer((__gm__ Element *)params.ptrOutReal);
        AscendC::GlobalTensor<Element> outImagGm;
        outImagGm.SetGlobalBuffer((__gm__ Element *)params.ptrOutImag);

        uint32_t eleNumPerBatch = params.layout.shape(0); // 16
        uint32_t iterCount = (eleNumPerBatch + COMPUTE_LENGTH - 1) / COMPUTE_LENGTH;

        for (uint32_t batchIdx = 0; batchIdx < params.batchSize; ++batchIdx) {
            for (uint32_t i = 0; i < STAGES; i++) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbEventList[i]);
            }

            uint32_t flagId = 0;

            for (uint32_t iterIdx = 0; iterIdx < iterCount; ++iterIdx) {
                // copy in
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(UbEventList[flagId]);
                uint32_t currComputeLength = iterIdx == iterCount - 1 ? eleNumPerBatch - iterIdx * COMPUTE_LENGTH : COMPUTE_LENGTH;
                uint32_t src1Offset = batchIdx * params.src1Stride + iterIdx * COMPUTE_LENGTH;
                uint32_t src2Offset = batchIdx * params.src2Stride + iterIdx * COMPUTE_LENGTH;
                uint32_t outOffset = batchIdx * params.outStride + iterIdx * COMPUTE_LENGTH;

                AscendC::DataCopyExtParams copyParams{(uint16_t)1, static_cast<uint32_t>(currComputeLength * sizeof(Element)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<Element> padParams{false, 0, 0, 0};

                AscendC::DataCopyPad(ubSrc1RealTensor[flagId], src1RealGm[src1Offset], copyParams, padParams);
                AscendC::DataCopyPad(ubSrc1ImagTensor[flagId], src1ImagGm[src1Offset], copyParams, padParams);
                AscendC::DataCopyPad(ubSrc2RealTensor[flagId], src2RealGm[src2Offset], copyParams, padParams);
                AscendC::DataCopyPad(ubSrc2ImagTensor[flagId], src2ImagGm[src2Offset], copyParams, padParams);

                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(UbEventList[flagId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(UbEventList[flagId]);

                complexMul(ubOutRealTensor[flagId], ubOutImagTensor[flagId],
                    ubSrc1RealTensor[flagId], ubSrc1ImagTensor[flagId],
                    ubSrc2RealTensor[flagId], ubSrc2ImagTensor[flagId],
                    ubTempTensor[flagId]);

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(UbEventList[flagId]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(UbEventList[flagId]);

                AscendC::DataCopyExtParams copyOutParams;
                copyOutParams.blockCount = 1;
                copyOutParams.blockLen = currComputeLength * sizeof(Element);
                copyOutParams.srcStride = 0;
                copyOutParams.dstStride = 0;
                copyOutParams.rsv = 0;

                AscendC::DataCopyPad(
                    outRealGm[outOffset],
                    ubOutRealTensor[flagId],
                    copyOutParams);

                AscendC::DataCopyPad(
                    outImagGm[outOffset],
                    ubOutImagTensor[flagId],
                    copyOutParams);

                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbEventList[flagId]);

                flagId = 1 - flagId;
            }

            for (uint32_t i = 0; i < STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(UbEventList[i]);
            }
        }
    };

private:
    Params params;
    AscendC::LocalTensor<Element> ubSrc1RealTensor[STAGES];
    AscendC::LocalTensor<Element> ubSrc1ImagTensor[STAGES];
    AscendC::LocalTensor<Element> ubSrc2RealTensor[STAGES];
    AscendC::LocalTensor<Element> ubSrc2ImagTensor[STAGES];
    AscendC::LocalTensor<Element> ubOutRealTensor[STAGES];
    AscendC::LocalTensor<Element> ubOutImagTensor[STAGES];
    AscendC::LocalTensor<Element> ubTempTensor[STAGES];

    AscendC::GlobalTensor<Element> src1RealGm;
    AscendC::GlobalTensor<Element> src1ImagGm;
    AscendC::GlobalTensor<Element> src2RealGm;
    AscendC::GlobalTensor<Element> src2ImagGm;
    AscendC::GlobalTensor<Element> outRealGm;
    AscendC::GlobalTensor<Element> outImagGm;

    VecCopyGmToUb vecCopyGmToUb;
    VecCopyUbToGm vecCopyUbToGm;

    TileElemWiseEpilogueComplexMul complexMul;

    int32_t UbEventList[STAGES];
};

}  // namespace Catlass::Epilogue::Block

#endif  // BLOCK_EPICLOGUE_COMPLEX_MUL_HPP