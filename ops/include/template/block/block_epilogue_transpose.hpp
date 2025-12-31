 /**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_EPICLOGUE_TRANSPOSE_HPP
#define BLOCK_EPICLOGUE_TRANSPOSE_HPP

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
    class ArchTag_,
    class DataType_,
    uint32_t STAGES_
>
class BlockEpilogueTranspose {
public:
    using ArchTag = ArchTag_;
    using Element = typename DataType_::Element;
    using ScaleLayout = typename DataType_::Layout;

    static constexpr uint32_t STAGES = STAGES_;

    struct Params {
        GM_ADDR ptrSrcReal;
        GM_ADDR ptrSrcImag;
        GM_ADDR ptrOutReal;
        GM_ADDR ptrOutImag;
        ScaleLayout layout; // [batch, innerBatch, radix, remain]

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GM_ADDR ptrSrcReal_,
            GM_ADDR ptrSrcImag_,
            GM_ADDR ptrOutReal_,
            GM_ADDR ptrOutImag_,
            ScaleLayout layout_)
            : ptrSrcReal(ptrSrcReal_), ptrSrcImag(ptrSrcImag_),
              ptrOutReal(ptrOutReal_), ptrOutImag(ptrOutImag_), layout(layout_) {}
    };

    // construct
    CATLASS_DEVICE
    BlockEpilogueTranspose(Arch::Resource<ArchTag>& resource, uint32_t ubByteStart = 0)
    {
        uint32_t numTensorList = 2;
        uint32_t tileSizeByByte = ArchTag::UB_SIZE / numTensorList / STAGES;
        tileSizeByByte = tileSizeByByte / 32 * 32;
        computeLength = tileSizeByByte / sizeof(Element);

        for (uint32_t i = 0; i < STAGES; ++i) {
            ubSrcRealTensor[i] = resource.ubBuf.template GetBufferByByte<Element>(ubByteStart);
            ubByteStart += tileSizeByByte;
            ubSrcImagTensor[i] = resource.ubBuf.template GetBufferByByte<Element>(ubByteStart);
            ubByteStart += tileSizeByByte;

            UbInEventList[i] = i;
        }
    }

    CATLASS_DEVICE
    ~BlockEpilogueTranspose()
    {}

    CATLASS_DEVICE
    void operator()(Params const& params)
    {
        AscendC::GlobalTensor<Element> srcRealGm;
        srcRealGm.SetGlobalBuffer((__gm__ Element *)params.ptrSrcReal);
        AscendC::GlobalTensor<Element> src1ImagGm;
        srcImagGm.SetGlobalBuffer((__gm__ Element *)params.ptrSrcImag);

        outRealGm.SetGlobalBuffer((__gm__ Element *)params.ptrOutReal);
        AscendC::GlobalTensor<Element> outImagGm;
        outImagGm.SetGlobalBuffer((__gm__ Element *)params.ptrOutImag);

        uint32_t eleNumPerBatch = params.layout.shape(2);
        uint32_t iterCount = (eleNumPerBatch + computeLength - 1) / computeLength;

        uint32_t innerBatchDim = params.layout.shape(0);
        uint32_t radixDim = params.layout.shape(1);
        uint32_t remainDim = params.layout.shape(2);

        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbInEventList[i]);
        }
        uint32_t flagId = 0;

        for (uint32_t innerIdx = 0; innerIdx < innerBatchDim; ++innerIdx) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(UbInEventList[flagId]);
            AscendC::DataCopyExtParams copyParams{(uint16_t)radixDim, static_cast<uint32_t>(eleNumPerBatch * sizeof(Element)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<Element> padParams{false, 0, 0, 0};

            uint32_t srcOffset = innerIdx * radixDim * remainDim;

            AscendC::DataCopyPad(ubSrcRealTensor[flagId], srcRealGm[srcOffset], copyParams, padParams);
            AscendC::DataCopyPad(ubSrcImagTensor[flagId], srcImagGm[srcOffset], copyParams, padParams);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(UbInEventList[flagId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(UbInEventList[flagId]);

            uint32_t dstOffset = innerIdx * remainDim;

            AscendC::DataCopyExtParams copyOutParams;
                copyOutParams.blockCount = (uint16_t)radixDim;
                copyOutParams.blockLen = static_cast<uint32_t>(eleNumPerBatch * sizeof(Element));
                copyOutParams.srcStride = 0;
                copyOutParams.dstStride = (innerBatchDim - 1) * remainDim * sizeof(Element);
                copyOutParams.rsv = 0;
            AscendC::DataCopyPad(outRealGm[dstOffset], ubSrcRealTensor[flagId], copyOutParams);
            AscendC::DataCopyPad(outImagGm[dstOffset], ubSrcImagTensor[flagId], copyOutParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbInEventList[flagId]);

            flagId = 1 - flagId;
        }

        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(UbInEventList[i]);
        }
    };

private:
    Params params;
    AscendC::LocalTensor<Element> ubSrcRealTensor[STAGES];
    AscendC::LocalTensor<Element> ubSrcImagTensor[STAGES];

    AscendC::GlobalTensor<Element> srcRealGm;
    AscendC::GlobalTensor<Element> srcImagGm;
    AscendC::GlobalTensor<Element> outRealGm;
    AscendC::GlobalTensor<Element> outImagGm;

    int32_t UbInEventList[STAGES];
    uint32_t computeLength;
};

}  // namespace Catlass::Epilogue::Block

#endif  // BLOCK_EPICLOGUE_TRANSPOSE_HPP