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

template <typename DTYPE = float, typename MM_OUT_DTYPE = float>
__aicore__ __attribute__((always_inline)) inline void load_matrix_zZ_dev(AscendC::LocalTensor<DTYPE> dst,
    AscendC::GlobalTensor<DTYPE> src, int32_t R, int32_t C, int32_t valid_row, int32_t valid_col, int32_t stride)
{
    constexpr int32_t R0 = 16;
    constexpr int32_t C0 = 32 / sizeof(DTYPE);
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

template <typename DTYPE = float, typename MM_OUT_DTYPE = float>
__aicore__ __attribute__((always_inline)) inline void base_matmul_single_core_dev(
    AscendC::GlobalTensor<DTYPE> matATensor, AscendC::GlobalTensor<DTYPE> matBTensor, AscendC::GlobalTensor<DTYPE> gm_c,
    AscendC::LocalTensor<DTYPE> l1_base_a_tensor, AscendC::LocalTensor<DTYPE> l1_base_b_tensor,
    AscendC::LocalTensor<DTYPE> l0a_base, AscendC::LocalTensor<DTYPE> l0b_base,
    AscendC::LocalTensor<MM_OUT_DTYPE> l0c_buf, uint32_t M, uint32_t N, uint32_t K, uint32_t leftRow, uint32_t leftCol,
    uint32_t rightRow, uint32_t rightCol, uint32_t outRow, uint32_t outCol, uint32_t trans_a, uint32_t trans_b,
    uint32_t loadA, uint32_t loadB)
{
    SetPadding(0);
    SetAtomicnone();

    int64_t L0AB_PINGPONG_BUFFER_LEN = 32 * 1024 / sizeof(DTYPE);  // 32KB
    int64_t L0C_PINGPONG_BUFFER_LEN = 64 * 1024 / sizeof(DTYPE);   // 64KB
    int64_t NUM_ELE_PERBLOCK = 32 / sizeof(DTYPE);                 // 每个block能放多少个float
    // CUBE的最小基块 （M，N，K）= 16 * 16 * 8
    int64_t CUBE_M0 = 16;
    int64_t CUBE_N0 = 16;
    int64_t CUBE_K0 = 32 / sizeof(DTYPE);
    int64_t CUBE_MATRIX_SIZE = CUBE_K0 * CUBE_N0;                 // 16 * 8
    int64_t L1_PINGPONG_BUFFER_LEN = 256 * 1024 / sizeof(DTYPE);  // 256KB
    int64_t UINT16_STRIDE_LIMIT = 65536;
    int32_t BLOCK_SIZE = 16;
    int32_t C0_SIZE = 32 / sizeof(DTYPE);

    int32_t batchSize, M0, N0, K0;
    batchSize = 1;

    M0 = 128;
    N0 = 128;
    K0 = 128;

    int32_t m_loop = (M + M0 - 1) / M0;
    int32_t n_loop = (N + N0 - 1) / N0;
    int32_t k_loop = (K + K0 - 1) / K0;
    int32_t loop = batchSize * m_loop * n_loop;

    int32_t ping_flag = 1;
    int32_t loop_ping_flag = 1;

    SET_FLAG(MTE1, MTE2, EVENT_ID0);
    SET_FLAG(MTE1, MTE2, EVENT_ID1);
    SET_FLAG(MTE1, MTE2, EVENT_ID2);
    SET_FLAG(MTE1, MTE2, EVENT_ID3);

    SET_FLAG(FIX, M, EVENT_ID0);

    for (int32_t loop_idx = 0; loop_idx < loop; loop_idx++) {

        int64_t batch_idx = loop_idx / (m_loop * n_loop);
        int64_t in_batch_idx = loop_idx % (m_loop * n_loop);  // 分块的id
        int64_t m_idx;
        int64_t n_idx;

        m_idx = loop_idx / n_loop;
        n_idx = loop_idx % n_loop;

        int64_t offset_a, offset_b;
        int64_t offset_c = m_idx * (int64_t)M0 * (int64_t)outCol + n_idx * (int64_t)N0;
        int32_t m_actual = (m_idx == (m_loop - 1)) ? (M - (int32_t)m_idx * M0) : M0;
        int32_t n_actual = (n_idx == (n_loop - 1)) ? (N - (int32_t)n_idx * N0) : N0;
        int32_t m_round = (m_actual + 15) / 16 * 16;
        int32_t n_round = (n_actual + 15) / 16 * 16;

        int32_t mn_max = m_round > n_round ? m_round : n_round;
        int32_t k_part_len = L0AB_PINGPONG_BUFFER_LEN / mn_max / 16 * 16;  // 128

        for (int32_t k_idx = 0; k_idx < k_loop; k_idx++) {
            if (trans_a) {
                // not implement yet
            } else {
                offset_a = m_idx * M0 * leftCol + k_idx * K0;
            }
            // not implement yet
            if (trans_b) {
            } else {
                offset_b = k_idx * K0 * rightCol + n_idx * N0;
            }

            int32_t k_actual = (k_idx == (k_loop - 1)) ? (K - k_idx * K0) : K0;
            int32_t k_round = (k_actual + 15) / 16 * 16;
            int32_t k_part_loop = (k_actual + k_part_len - 1) / k_part_len;

            auto l1_buf_a = ping_flag ? l1_base_a_tensor : l1_base_a_tensor[L1_PINGPONG_BUFFER_LEN];
            auto l1_buf_b = ping_flag ? l1_base_b_tensor : l1_base_b_tensor[L1_PINGPONG_BUFFER_LEN];
            auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

            // *** load matrix A to L1
            WAIT_FLAG(MTE1, MTE2, event_id);

            if (m_actual == 1) {
                AscendC::DataCopy(l1_buf_a,
                    matATensor[offset_a],
                    AscendC::DataCopyParams(1,               // nBurst
                        (k_actual + C0_SIZE - 1) / C0_SIZE,  // lenBurst
                        0,                                   // srcGap
                        0                                    // dstGap
                        ));
            } else {
                load_matrix_zZ_dev<DTYPE, MM_OUT_DTYPE>(
                    l1_buf_a, matATensor[offset_a], M0, K0, m_actual, k_actual, leftCol);
            }

            SET_FLAG(MTE2, MTE1, event_id);

            // *** load matrix B to L1
            WAIT_FLAG(MTE1, MTE2, event_id + 2);

            load_matrix_zZ_dev<DTYPE, MM_OUT_DTYPE>(
                l1_buf_b, matBTensor[offset_b], K0, N0, k_actual, n_actual, rightCol);

            SET_FLAG(MTE2, MTE1, event_id + 2);

            SET_FLAG(M, MTE1, EVENT_ID0);
            SET_FLAG(M, MTE1, EVENT_ID1);

            for (int32_t k_part_idx = 0; k_part_idx < k_part_loop; k_part_idx++) {
                int32_t k0_round = (k_part_idx < k_part_loop - 1) ? k_part_len : k_round - k_part_idx * k_part_len;
                int32_t k0_actual = (k_part_idx < k_part_loop - 1) ? k_part_len : k_actual - k_part_idx * k_part_len;

                auto mte1_mad_ping_flag = 1 - k_part_idx % 2;
                auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
                auto l0a_buf = l0a_base[(k_part_idx % 2) * L0AB_PINGPONG_BUFFER_LEN];
                auto l0b_buf = l0b_base[(k_part_idx % 2) * L0AB_PINGPONG_BUFFER_LEN];

                // *** load matrix A from L1 to L0A
                if (k_part_idx == 0) {
                    WAIT_FLAG(MTE2, MTE1, event_id);
                }
                WAIT_FLAG(M, MTE1, mte1_mad_event_id);
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

                    } else {
                        auto l1_src_a = l1_buf_a[k_part_idx * k_part_len * BLOCK_SIZE];
                        for (int i = 0; i < m_round / BLOCK_SIZE; i++) {
                            AscendC::LoadData(l0a_buf[i * k0_round * BLOCK_SIZE],
                                l1_src_a[i * BLOCK_SIZE * K0],
                                AscendC::LoadData2dParams(0,  // baseIdx
                                    k0_round / C0_SIZE,       // repeat
                                    1,                        // srcStride
                                    0,                        // sid
                                    0,                        // dstGap
                                    false,                    // transpose
                                    inc                       // addr_cal_mode_t
                                    ));
                        }
                    }
                }
                if (k_part_idx == k_part_loop - 1) {
                    SET_FLAG(MTE1, MTE2, event_id);
                }

                // *** load matrix B from L1 to L0B
                if (k_part_idx == 0) {
                    WAIT_FLAG(MTE2, MTE1, event_id + 2);
                }

                if (trans_b) {

                } else {
                    auto l1_src_b = l1_buf_b[k_part_idx * k_part_len * N0];
                    int32_t dstGap = sizeof(DTYPE) == 2 ? n_round / BLOCK_SIZE - 1 : 2 * n_round / BLOCK_SIZE - 1;
                    int32_t l1Offset = sizeof(DTYPE) == 2 ? 1 : 2;
                    for (int i = 0; i < n_round / BLOCK_SIZE; i++) {
                        AscendC::LoadDataWithTranspose(l0b_buf[i * CUBE_MATRIX_SIZE],
                            l1_src_b[i * l1Offset * CUBE_MATRIX_SIZE],
                            AscendC::LoadData2dTransposeParams(0,  // indexID
                                k0_round / BLOCK_SIZE,             // repeat
                                N0 / BLOCK_SIZE,                   // srcStride
                                // 2 * n_round / BLOCK_SIZE - 1,  // dstGap
                                dstGap,                    // dstGap
                                n_round / BLOCK_SIZE - 1,  // dstFracStride
                                inc                        // addrmode
                                ));
                    }
                }

                if (k_part_idx == k_part_loop - 1) {
                    SET_FLAG(MTE1, MTE2, event_id + 2);
                }

                SET_FLAG(MTE1, M, mte1_mad_event_id);
                WAIT_FLAG(MTE1, M, mte1_mad_event_id);

                bool init_c = (k_idx == 0 && k_part_idx == 0);
                if (init_c) {
                    WAIT_FLAG(FIX, M, EVENT_ID0);
                }

                if (M != 1 && m_actual == 1 && trans_a) {

                } else {
                    AscendC::MmadParams mmParam = AscendC::MmadParams(m_actual, n_actual, k0_actual, 0, false, init_c);
                    mmParam.kDirectionAlign = true;

                    AscendC::Mmad(l0c_buf, l0a_buf, l0b_buf, mmParam);
                }

                PIPE_BARRIER(M);
                SET_FLAG(M, MTE1, mte1_mad_event_id);
            }

            WAIT_FLAG(M, MTE1, EVENT_ID0);
            WAIT_FLAG(M, MTE1, EVENT_ID1);
            ping_flag = 1 - ping_flag;
        }

        SET_FLAG(M, FIX, EVENT_ID0);
        WAIT_FLAG(M, FIX, EVENT_ID0);

        l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, DTYPE, float>(gm_c[offset_c],
            l0c_buf,
            m_actual,  // MSize
            n_actual,  // NSize
            m_round,   // srcStride
            outCol     // dstStride_dst_D
        );

        loop_ping_flag = 1 - loop_ping_flag;
        SET_FLAG(FIX, M, EVENT_ID0);
    }

    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);

    WAIT_FLAG(FIX, M, EVENT_ID0);

    PIPE_BARRIER(ALL);
}
