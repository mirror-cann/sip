/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "kernel_operator.h"

// ----------------- cube parameters ---------------------
#ifdef __DAV_C220_CUBE__
constexpr int64_t AIV_IN_GROUP_CORE_NUM = 2;
constexpr int64_t L0AB_PINGPONG_BUFFER_LEN = 32 * 1024 / sizeof(half);  // 32KB
constexpr int64_t L0C_PINGPONG_BUFFER_LEN = 64 * 1024 / sizeof(half);   // 64KB
constexpr int64_t NUM_ELE_PERBLOCK = 32 / sizeof(half);                 // 每个block能放多少个float
// half CUBE的最小基块 （M，N，K）= 16 * 16 * 8
constexpr int64_t CUBE_M0 = 16;
constexpr int64_t CUBE_N0 = 16;
constexpr int64_t CUBE_K0 = 32 / sizeof(half);
constexpr int64_t CUBE_MATRIX_SIZE = CUBE_K0 * CUBE_N0;                // 16 * 8
constexpr int64_t L1_PINGPONG_BUFFER_LEN = 256 * 1024 / sizeof(half);  // 256KB
constexpr int64_t UINT16_STRIDE_LIMIT = 65536;
constexpr int32_t R0_SIZE = 16;
constexpr int32_t C0_SIZE = 32 / sizeof(half);

#endif
// ----------------- cube parameters end ---------------------

#ifdef __DAV_C220_CUBE__
template <typename DTYPE>
__aicore__ __attribute__((always_inline)) inline void load_matrix_zZ_dev(AscendC::LocalTensor<DTYPE> dst,
    AscendC::GlobalTensor<DTYPE> src, int32_t R, int32_t C, int32_t valid_row, int32_t valid_col, int32_t stride)
{
    constexpr int32_t R0 = 16;
    constexpr int32_t C0 = 32 / sizeof(half);
    constexpr int STRIDE_LIMIT = 65536;

    int64_t srcNdStride = R0 * stride;
    int64_t srcNStride = stride;

    if (srcNdStride < STRIDE_LIMIT) {
        int32_t ndNum = valid_row / R0;
        int32_t remains = valid_row % R0;

        if (ndNum > 0) {
            AscendC::DataCopy(dst,
                src,
                AscendC::Nd2NzParams(ndNum,  // ndNum
                    R0,                      // nValue
                    valid_col,               // dValue
                    R0 * stride,             // srcNdMatrixStride
                    srcNStride,              // srcDValue
                    R0,                      // dstNzC0Stride
                    1,                       // dstNzNStride
                    R0 * C)                  // dstNzMatrixStride
            );
        }
        if (remains > 0) {
            AscendC::DataCopy(dst[ndNum * R0 * C],
                src[ndNum * R0 * stride],
                AscendC::Nd2NzParams(1,  // ndNum
                    remains,             // nValue
                    valid_col,           // dValue
                    0,                   // srcNdMatrixStride
                    srcNStride,          // srcDValue
                    R0,                  // dstNzC0Stride
                    1,                   // dstNzNStride
                    0)                   // dstNzMatrixStride
            );
        }
    } else if (srcNStride < STRIDE_LIMIT) {
        int32_t ndNum = valid_row / R0;
        int32_t remains = valid_row % R0;
        for (int32_t i = 0; i < ndNum; i++) {
            AscendC::DataCopy(dst[i * R0 * C],
                src[i * R0 * stride],
                AscendC::Nd2NzParams(1,  // ndNum
                    R0,                  // nValue
                    valid_col,           // dValue
                    0,                   // srcNdMatrixStride
                    srcNStride,          // srcDValue
                    R0,                  // dstNzC0Stride
                    1,                   // dstNzNStride
                    0)                   // dstNzMatrixStride
            );
        }
        if (remains > 0) {
            AscendC::DataCopy(dst[ndNum * R0 * C],
                src[ndNum * R0 * stride],
                AscendC::Nd2NzParams(1,  // ndNum
                    remains,             // nValue
                    valid_col,           // dValue
                    0,                   // srcNdMatrixStride
                    srcNStride,          // srcDValue
                    R0,                  // dstNzC0Stride
                    1,                   // dstNzNStride
                    0)                   // dstNzMatrixStride
            );
        }
    } else {
        for (int32_t i = 0; i < valid_row; i++) {
            int32_t idxR0 = i / R0;
            int32_t idxInR0 = i % R0;

            AscendC::DataCopy(dst[idxR0 * R0 * C + idxInR0 * C0],
                src[i * stride],
                AscendC::Nd2NzParams(1,  // ndNum
                    1,                   // nValue
                    valid_col,           // dValue
                    0,                   // srcNdMatrixStride
                    0,                   // srcDValue
                    R0,                  // dstNzC0Stride
                    0,                   // dstNzNStride
                    0)                   // dstNzMatrixStride
            );
        }
    }
}

// R must be a multiple of 2
template <typename DTYPE>
__aicore__ __attribute__((always_inline)) inline void load_right_matrix_zZ_dev(AscendC::LocalTensor<DTYPE> dst,
    AscendC::GlobalTensor<DTYPE> firstSrc, AscendC::GlobalTensor<DTYPE> secondSrc, int32_t R, int32_t C,
    int32_t valid_row, int32_t valid_col, int32_t stride)
{
    constexpr int32_t R0 = 16;
    constexpr int32_t C0 = 32 / sizeof(half);

    constexpr int32_t srcR0 = R0 / 2;
    int32_t srcValidRow = valid_row / 2;

    constexpr int STRIDE_LIMIT = 65536;

    int32_t ndNum = srcValidRow / srcR0;
    int32_t remains = srcValidRow % srcR0;

    if (ndNum > 0) {
        AscendC::DataCopy(dst,
            firstSrc,
            AscendC::Nd2NzParams(ndNum,  // ndNum
                srcR0,                   // nValue
                valid_col,               // dValue
                srcR0 * stride,          // srcNdMatrixStride
                stride,                  // srcDValue
                R0,                      // dstNzC0Stride
                2,                       // dstNzNStride
                R0 * C)                  // dstNzMatrixStride
        );
        AscendC::DataCopy(dst[C0],
            secondSrc,
            AscendC::Nd2NzParams(ndNum,  // ndNum
                srcR0,                   // nValue
                valid_col,               // dValue
                srcR0 * stride,          // srcNdMatrixStride
                stride,                  // srcDValue
                R0,                      // dstNzC0Stride
                2,                       // dstNzNStride
                R0 * C)                  // dstNzMatrixStride
        );
    }

    if (remains > 0) {
        AscendC::DataCopy(dst[ndNum * R0 * C],
            firstSrc[ndNum * srcR0 * stride],
            AscendC::Nd2NzParams(1,  // ndNum
                remains,             // nValue
                valid_col,           // dValue
                0,                   // srcNdMatrixStride
                stride,              // srcDValue
                R0,                  // dstNzC0Stride
                2,                   // dstNzNStride
                0)                   // dstNzMatrixStride
        );
        AscendC::DataCopy(dst[ndNum * R0 * C + C0],
            secondSrc[ndNum * srcR0 * stride],
            AscendC::Nd2NzParams(1,  // ndNum
                remains,             // nValue
                valid_col,           // dValue
                0,                   // srcNdMatrixStride
                stride,              // srcDValue
                R0,                  // dstNzC0Stride
                2,                   // dstNzNStride
                0)                   // dstNzMatrixStride
        );
    }
}

template <typename DTYPE>
__aicore__ __attribute__((always_inline)) inline void load_a_to_l1(AscendC::LocalTensor<DTYPE> l1_buf_a,
    AscendC::GlobalTensor<DTYPE> matATensor, int64_t offset_a, int64_t M, int64_t M0, int64_t K0, int64_t m_actual,
    int64_t k_actual, int64_t K, int64_t trans_a)
{
    if (trans_a) {
        assert(false, "Invalid branch.");
        if (M == 1) {
            assert(false, "Invalid branch.");
        } else {
            assert(false, "Invalid branch.");
        }
    } else {
        if (m_actual == 1) {
            AscendC::DataCopy(l1_buf_a,
                matATensor[offset_a],
                AscendC::DataCopyParams(1,               // nBurst
                    (k_actual + C0_SIZE - 1) / C0_SIZE,  // lenBurst
                    0,                                   // srcGap
                    0                                    // dstGap
                    ));
        } else {
            load_matrix_zZ_dev(l1_buf_a, matATensor[offset_a], M0, K0, m_actual, k_actual, K);
        }
    }
}

template <typename DTYPE>
__aicore__ __attribute__((always_inline)) inline void load_b_to_l1(AscendC::LocalTensor<DTYPE> l1_buf_b,
    AscendC::GlobalTensor<DTYPE> firstMatBTensor, AscendC::GlobalTensor<DTYPE> secondMatBTensor, int64_t offset_b,
    int64_t K0, int64_t N0, int64_t k_actual, int64_t n_actual, int64_t N, int64_t trans_b)
{
    if (trans_b) {
        assert(false, "Invalid branch.");
    } else {
        load_right_matrix_zZ_dev(
            l1_buf_b, firstMatBTensor[offset_b], secondMatBTensor[offset_b], K0, N0, k_actual, n_actual, N);
    }
}

template <typename DTYPE>
__aicore__ __attribute__((always_inline)) inline void load_l1_to_l0a(AscendC::LocalTensor<DTYPE> l0a_buf,
    AscendC::LocalTensor<DTYPE> l1_buf_a, int64_t k0_round, int64_t m_round, int64_t k_part_idx, int64_t k_part_len,
    int64_t m_actual, int64_t M, int64_t K0, int64_t trans_a, int64_t inc)
{
    if (M == 1 || m_actual == 1 && !trans_a) {
        AscendC::LoadData(l0a_buf,
            l1_buf_a[k_part_idx * k_part_len],
            AscendC::LoadData2dParams(0,                               // baseIdx
                (k0_round + CUBE_MATRIX_SIZE - 1) / CUBE_MATRIX_SIZE,  // repeat
                1,                                                     // srcStride
                0,                                                     // sid
                0,                                                     // dstStride
                false,                                                 // transpose
                0                                                      // addr_cal_mode_t
                ));
    } else {
        if (trans_a) {
            assert(false, "Invalid branch.");
        } else {
            auto l1_src_a = l1_buf_a[k_part_idx * k_part_len * R0_SIZE];
            for (int i = 0; i < m_round / R0_SIZE; i++) {
                AscendC::LoadData(l0a_buf[i * k0_round * R0_SIZE],
                    l1_src_a[i * R0_SIZE * K0],
                    AscendC::LoadData2dParams(0,  // baseIdx
                        k0_round / C0_SIZE,       // repeat
                        1,                        // srcStride
                        0,                        // sid
                        0,                        // dstStride
                        false,                    // transpose
                        inc                       // addr_cal_mode_t
                        ));
            }
        }
    }
}

template <typename DTYPE>
__aicore__ __attribute__((always_inline)) inline void load_l1_to_l0b(AscendC::LocalTensor<DTYPE> l0b_buf,
    AscendC::LocalTensor<DTYPE> l1_buf_b, int64_t k0_round, int64_t n_round, int64_t k_part_idx, int64_t k_part_len,
    int64_t N0, int64_t trans_b, int64_t inc)
{
    if (trans_b) {
        assert(false, "Invalid branch.");
    } else {
        auto l1_src_b = l1_buf_b[k_part_idx * k_part_len * N0];
        for (int i = 0; i < n_round / R0_SIZE; i++) {
            AscendC::LoadDataWithTranspose(l0b_buf[i * CUBE_MATRIX_SIZE],
                l1_src_b[i * 1 * CUBE_MATRIX_SIZE],
                AscendC::LoadData2dTransposeParams(0,  // indexID
                    k0_round / R0_SIZE,                // repeat
                    N0 / R0_SIZE,                      // srcStride
                    n_round / R0_SIZE - 1,             // dstStride
                    0,                                 // dstFracStride
                    inc                                // addrmode
                    ));
        }
    }
}

template <typename DTYPE>
__aicore__ __attribute__((always_inline)) inline void do_mmad(AscendC::LocalTensor<float> l0c_buf_tensor,
    AscendC::LocalTensor<DTYPE> l0a_buf, AscendC::LocalTensor<DTYPE> l0b_buf, int64_t m_actual, int64_t n_actual,
    int64_t k0_actual, int64_t M, int64_t trans_a, int64_t init_c)
{
    if (M != 1 && m_actual == 1 && trans_a) {
        assert(false, "Invalid branch.");
    } else {
        AscendC::MmadParams mmadParams = AscendC::MmadParams(m_actual,
            n_actual,
            k0_actual,
            0,      // unitFlag
            false,  // cmatrixSource
            init_c  // cmatrixInitVal
        );

        mmadParams.kDirectionAlign = true;

        AscendC::Mmad(l0c_buf_tensor, l0a_buf, l0b_buf, mmadParams);
    }
}

template <typename DTYPE>
__aicore__ __attribute__((always_inline)) inline void load_l0c_to_c(AscendC::GlobalTensor<DTYPE> matCTensor,
    AscendC::LocalTensor<float> l0c_buf_tensor, int64_t offset_c, int64_t m_actual, int64_t n_actual, int64_t m_round,
    int64_t N)
{
    l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, DTYPE, float>(matCTensor[offset_c],
        l0c_buf_tensor,
        m_actual,  // MSize
        n_actual,  // NSize
        m_round,   // srcStride
        N          // dstStride_dst_D
    );
}

template <typename DTYPE>
__aicore__ __attribute__((always_inline)) inline void batch_base_matmul_cfp32_dev(
    AscendC::GlobalTensor<DTYPE> matATensor, AscendC::GlobalTensor<DTYPE> firstMatBTensor,
    AscendC::GlobalTensor<DTYPE> secondMatBTensor, AscendC::GlobalTensor<DTYPE> matCTensor,
    AscendC::LocalTensor<DTYPE> l1_base_a_tensor, AscendC::LocalTensor<DTYPE> l1_base_b_tensor,
    AscendC::LocalTensor<DTYPE> l0a_base_tensor, AscendC::LocalTensor<DTYPE> l0b_base_tensor,
    AscendC::LocalTensor<float> l0c_buf_tensor, uint32_t B, uint32_t M, uint32_t K, uint32_t N, uint32_t trans_a,
    uint32_t trans_b)
{
    AscendC::SetLoadDataPaddingValue<DTYPE>(0.0);
    AscendC::SetAtomicNone();

    int32_t batchSize, M0, N0, K0;
    batchSize = B;

    M0 = 128;
    N0 = 128;
    K0 = 128;

    int64_t srcK = K / 2;
    int64_t srcK0 = K0 / 2;

    int32_t m_loop = (M + M0 - 1) / M0;
    int32_t n_loop = (N + N0 - 1) / N0;
    int32_t k_loop = (K + K0 - 1) / K0;
    int32_t loop = batchSize * m_loop * n_loop;

    int32_t ping_flag = 1;
    int32_t loop_ping_flag = 1;

    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);

    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    int32_t blockNum = get_block_num();

    for (int32_t loop_idx = 0; loop_idx < loop; loop_idx++) {

        int64_t batch_idx = loop_idx / (m_loop * n_loop);     // batch间的分块id
        int64_t in_batch_idx = loop_idx % (m_loop * n_loop);  // batch内的分块id
        int64_t m_idx;
        int64_t n_idx;

        m_idx = in_batch_idx / n_loop;
        n_idx = in_batch_idx % n_loop;

        int64_t offset_a, offset_b;
        int64_t offset_c = batch_idx * (int64_t)M * (int64_t)N + m_idx * (int64_t)M0 * (int64_t)N + n_idx * (int64_t)N0;
        int32_t m_actual = (m_idx == (m_loop - 1)) ? (M - (int32_t)m_idx * M0) : M0;  // sub compute m
        int32_t n_actual = (n_idx == (n_loop - 1)) ? (N - (int32_t)n_idx * N0) : N0;  // sub compute n
        int32_t m_round = (m_actual + 15) / 16 * 16;
        int32_t n_round = (n_actual + 15) / 16 * 16;

        int32_t mn_max = m_round > n_round ? m_round : n_round;
        int32_t k_part_len = L0AB_PINGPONG_BUFFER_LEN / mn_max / 16 * 16;

        for (int64_t k_idx = 0; k_idx < k_loop; k_idx++) {
            if (trans_a) {
                assert(false, "Invalid branch.");
            } else {
                offset_a = batch_idx * M * K + m_idx * M0 * K + k_idx * K0;
            }

            if (trans_b) {
                assert(false, "Invalid branch.");
            } else {
                offset_b = batch_idx * srcK * N + k_idx * srcK0 * N + n_idx * N0;
            }

            int32_t k_actual = (k_idx == (k_loop - 1)) ? (K - k_idx * K0) : K0;
            int32_t k_round = (k_actual + 15) / 16 * 16;
            int32_t k_part_loop = (k_actual + k_part_len - 1) / k_part_len;

            auto l1_buf_a = ping_flag ? l1_base_a_tensor : l1_base_a_tensor[L1_PINGPONG_BUFFER_LEN];
            auto l1_buf_b = ping_flag ? l1_base_b_tensor : l1_base_b_tensor[L1_PINGPONG_BUFFER_LEN];
            auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

            // *** load matrix A to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(event_id);
            load_a_to_l1(l1_buf_a, matATensor, offset_a, M, M0, K0, m_actual, k_actual, K, trans_a);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(event_id);

            // *** load matrix B to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(event_id + 2);
            load_b_to_l1(l1_buf_b, firstMatBTensor, secondMatBTensor, offset_b, K0, N0, k_actual, n_actual, N, trans_b);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(event_id + 2);

            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);

            for (int32_t k_part_idx = 0; k_part_idx < k_part_loop; k_part_idx++) {
                int32_t k0_round = (k_part_idx < k_part_loop - 1) ? k_part_len : k_round - k_part_idx * k_part_len;
                int32_t k0_actual = (k_part_idx < k_part_loop - 1) ? k_part_len : k_actual - k_part_idx * k_part_len;

                auto mte1_mad_ping_flag = 1 - k_part_idx % 2;
                auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
                auto l0a_buf = l0a_base_tensor[(k_part_idx % 2) * L0AB_PINGPONG_BUFFER_LEN];
                auto l0b_buf = l0b_base_tensor[(k_part_idx % 2) * L0AB_PINGPONG_BUFFER_LEN];

                // *** load matrix A from L1 to L0A
                if (k_part_idx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(event_id);
                }
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mte1_mad_event_id);
                load_l1_to_l0a(
                    l0a_buf, l1_buf_a, k0_round, m_round, k_part_idx, k_part_len, m_actual, M, K0, trans_a, inc);
                if (k_part_idx == k_part_loop - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(event_id);
                }

                // *** load matrix B from L1 to L0B
                if (k_part_idx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(event_id + 2);
                }

                load_l1_to_l0b(l0b_buf, l1_buf_b, k0_round, n_round, k_part_idx, k_part_len, N0, trans_b, inc);

                if (k_part_idx == k_part_loop - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(event_id + 2);
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(mte1_mad_event_id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(mte1_mad_event_id);

                bool init_c = (k_idx == 0 && k_part_idx == 0);
                if (init_c) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
                }

                do_mmad(l0c_buf_tensor, l0a_buf, l0b_buf, m_actual, n_actual, k0_actual, M, trans_a, init_c);

                AscendC::PipeBarrier<PIPE_M>();
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mte1_mad_event_id);
            }

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            ping_flag = 1 - ping_flag;
        }

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);

        // copy from L0C to gm
        load_l0c_to_c(matCTensor, l0c_buf_tensor, offset_c, m_actual, n_actual, m_round, N);

        loop_ping_flag = 1 - loop_ping_flag;
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);

    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);

    AscendC::PipeBarrier<PIPE_ALL>();
}

#endif
