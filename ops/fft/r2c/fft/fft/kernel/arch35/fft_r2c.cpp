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
#include "simt_api/asc_simt.h"

using namespace AscendC;

constexpr uint32_t VF_MAX_THREAD_NUM = 256;
constexpr int32_t MAX_FFT_STAGES = 32;

__simt_callee__ inline void ComplexMul(
    float ar, float ai, float br, float bi, float &cr, float &ci)
{
    cr = ar * br - ai * bi;
    ci = ar * bi + ai * br;
}

__simt_callee__ inline int32_t FindRadix(int64_t n)
{
    if (n % 2 == 0) return 2;
    if (n % 3 == 0) return 3;
    if (n % 5 == 0) return 5;
    if (n % 7 == 0) return 7;
    return 0;
}

__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) __aicore__ void FftR2CPackRealToComplex(
    __gm__ float * __restrict__ gm_input,
    __gm__ float * __restrict__ gm_workspace,
    int64_t batchSize,
    int64_t fftN,
    int64_t wsOffset)
{
    int64_t N_half = fftN / 2;
    int64_t totalComplex = batchSize * N_half;

    int64_t tid = static_cast<int64_t>(Simt::GetThreadIdx<0>()) +
                  static_cast<int64_t>(Simt::GetBlockIdx()) * static_cast<int64_t>(Simt::GetThreadNum<0>());
    int64_t stride = static_cast<int64_t>(Simt::GetBlockNum()) * static_cast<int64_t>(Simt::GetThreadNum<0>());

    int64_t wsFloatOffset = wsOffset / sizeof(float);

    for (int64_t idx = tid; idx < totalComplex; idx += stride) {
        int64_t b = idx / N_half;
        int64_t k = idx % N_half;

        int64_t inIdxReal = b * fftN + 2 * k;
        int64_t inIdxImag = b * fftN + 2 * k + 1;

        int64_t outFloatIdx = wsFloatOffset + (b * N_half + k) * 2;
        gm_workspace[outFloatIdx]     = gm_input[inIdxReal];
        gm_workspace[outFloatIdx + 1] = gm_input[inIdxImag];
    }
}

__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) __aicore__ void FftR2CStockhamForwardStage(
    __gm__ float * __restrict__ inputBuf,
    __gm__ float * __restrict__ outputBuf,
    __gm__ float * __restrict__ dftMat,
    __gm__ float * __restrict__ twMat,
    int32_t radix,
    int64_t M,
    int64_t currentBatch,
    int32_t step,
    int32_t radixListLen)
{
    int64_t tid = static_cast<int64_t>(Simt::GetThreadIdx<0>()) +
                  static_cast<int64_t>(Simt::GetBlockIdx()) * static_cast<int64_t>(Simt::GetThreadNum<0>());
    int64_t stride = static_cast<int64_t>(Simt::GetBlockNum()) * static_cast<int64_t>(Simt::GetThreadNum<0>());

    int64_t totalOutputComplex = currentBatch * radix * M;
    
    for (int64_t outIdx = tid; outIdx < totalOutputComplex; outIdx += stride) {
        int64_t batchIdx = outIdx / (radix * M);
        int64_t rem = outIdx - batchIdx * (radix * M);
        int64_t u = rem / M;
        int64_t n2 = rem - u * M;

        float resultRe = 0.0f;
        float resultIm = 0.0f;

        #define DFT_TERM(v) do { \
            float wrRe = dftMat[(2 * u) * radix + (v)]; \
            float wrIm = dftMat[(2 * u + 1) * radix + (v)]; \
            int64_t xFloatIdx = (batchIdx * radix * M + (v) * M + n2) * 2; \
            float xRe = inputBuf[xFloatIdx]; \
            float xIm = inputBuf[xFloatIdx + 1]; \
            float mulRe, mulIm; \
            ComplexMul(wrRe, wrIm, xRe, xIm, mulRe, mulIm); \
            resultRe += mulRe; \
            resultIm += mulIm; \
        } while(0)

        if (radix == 2) {
            DFT_TERM(0); DFT_TERM(1);
        } else if (radix == 3) {
            DFT_TERM(0); DFT_TERM(1); DFT_TERM(2);
        } else if (radix == 5) {
            DFT_TERM(0); DFT_TERM(1); DFT_TERM(2); DFT_TERM(3); DFT_TERM(4);
        } else if (radix == 7) {
            DFT_TERM(0); DFT_TERM(1); DFT_TERM(2); DFT_TERM(3); DFT_TERM(4); DFT_TERM(5); DFT_TERM(6);
        }
        
        #undef DFT_TERM

        if (step < radixListLen - 1) {
            float tRe = twMat[(2 * u) * M + n2];
            float tIm = twMat[(2 * u + 1) * M + n2];

            float tmpRe, tmpIm;
            ComplexMul(resultRe, resultIm, tRe, tIm, tmpRe, tmpIm);
            resultRe = tmpRe;
            resultIm = tmpIm;
        }

        int64_t outComplexIdx = (batchIdx * radix + u) * M + n2;
        int64_t outFloatIdx = outComplexIdx * 2;
        outputBuf[outFloatIdx]     = resultRe;
        outputBuf[outFloatIdx + 1] = resultIm;
    }
}

__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) __aicore__ void FftR2CTransposeStage(
    __gm__ float * __restrict__ inputBuf,
    __gm__ float * __restrict__ outputBuf,
    int32_t radix,
    int64_t M,
    int64_t currentBatch)
{
    int64_t tid = static_cast<int64_t>(Simt::GetThreadIdx<0>()) +
                  static_cast<int64_t>(Simt::GetBlockIdx()) * static_cast<int64_t>(Simt::GetThreadNum<0>());
    int64_t stride = static_cast<int64_t>(Simt::GetBlockNum()) * static_cast<int64_t>(Simt::GetThreadNum<0>());
    int64_t totalComplex = currentBatch * radix * M;

    for (int64_t outIdx = tid; outIdx < totalComplex; outIdx += stride) {
        int64_t batchIdx = outIdx / (M * radix);
        int64_t rem = outIdx - batchIdx * (M * radix);
        int64_t n2 = rem / radix;
        int64_t k1 = rem - n2 * radix;

        int64_t inComplexIdx = batchIdx * radix * M + k1 * M + n2;
        int64_t inFloatIdx = inComplexIdx * 2;

        float re = inputBuf[inFloatIdx];
        float im = inputBuf[inFloatIdx + 1];

        int64_t outFloatIdx = outIdx * 2;
        outputBuf[outFloatIdx]     = re;
        outputBuf[outFloatIdx + 1] = im;
    }
}

__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) __aicore__ void FftR2CPostProcess(
    __gm__ float * __restrict__ gm_workspace,
    __gm__ float * __restrict__ gm_tw_post,
    __gm__ float * __restrict__ gm_output,
    int64_t batchSize,
    int64_t fftN,
    int64_t wsOffset)
{
    int64_t N_half = fftN / 2;
    int64_t D = N_half + 1;

    int64_t tid = static_cast<int64_t>(Simt::GetThreadIdx<0>()) +
                  static_cast<int64_t>(Simt::GetBlockIdx()) * static_cast<int64_t>(Simt::GetThreadNum<0>());
    int64_t stride = static_cast<int64_t>(Simt::GetBlockNum()) * static_cast<int64_t>(Simt::GetThreadNum<0>());

    int64_t wsFloatOffset = wsOffset / sizeof(float);

    for (int64_t idx = tid; idx < batchSize * D; idx += stride) {
        int64_t b = idx / D;
        int64_t k = idx % D;

        int64_t outFloatIdx = (b * D + k) * 2;

        if (k == 0) {
            int64_t z0FloatIdx = wsFloatOffset + (b * N_half) * 2;
            float z0Re = gm_workspace[z0FloatIdx];
            float z0Im = gm_workspace[z0FloatIdx + 1];
            gm_output[outFloatIdx]     = z0Re + z0Im;
            gm_output[outFloatIdx + 1] = 0.0f;
        } else if (k == N_half) {
            int64_t z0FloatIdx = wsFloatOffset + (b * N_half) * 2;
            float z0Re = gm_workspace[z0FloatIdx];
            float z0Im = gm_workspace[z0FloatIdx + 1];
            gm_output[outFloatIdx]     = z0Re - z0Im;
            gm_output[outFloatIdx + 1] = 0.0f;
        } else {
            int64_t zkFloatIdx = wsFloatOffset + (b * N_half + k) * 2;
            float zkRe = gm_workspace[zkFloatIdx];
            float zkIm = gm_workspace[zkFloatIdx + 1];

            int64_t znkFloatIdx = wsFloatOffset + (b * N_half + (N_half - k)) * 2;
            float znkRe =  gm_workspace[znkFloatIdx];
            float znkIm = -gm_workspace[znkFloatIdx + 1];

            float wkRe = gm_tw_post[k * 2];
            float wkIm = gm_tw_post[k * 2 + 1];

            float X_e_Re = (zkRe + znkRe) / 2.0f;
            float X_e_Im = (zkIm + znkIm) / 2.0f;

            float X_o_Re = (zkIm - znkIm) / 2.0f;
            float X_o_Im = (znkRe - zkRe) / 2.0f;

            float mulRe, mulIm;
            ComplexMul(wkRe, wkIm, X_o_Re, X_o_Im, mulRe, mulIm);

            gm_output[outFloatIdx]     = X_e_Re + mulRe;
            gm_output[outFloatIdx + 1] = X_e_Im + mulIm;
        }
    }
}

class FftR2CKernel {
public:
    __aicore__ inline FftR2CKernel() {}

    __aicore__ inline void Init(
        __gm__ float * __restrict__ gm_input,
        __gm__ float * __restrict__ gm_dft_matrix_array,
        __gm__ float * __restrict__ gm_tw_matrix_array,
        __gm__ float * __restrict__ gm_tw_post_process,
        __gm__ float * __restrict__ radix_list,
        __gm__ float * __restrict__ gm_output,
        __gm__ float * __restrict__ gm_workspace,
        __gm__ uint8_t * __restrict__ gm_tiling_para);

    __aicore__ inline void Process();

private:
    __gm__ float * __restrict__ gm_input_;
    __gm__ float * __restrict__ gm_dft_matrix_array_;
    __gm__ float * __restrict__ gm_tw_matrix_array_;
    __gm__ float * __restrict__ gm_tw_post_process_;
    __gm__ float * __restrict__ gm_output_;
    __gm__ float * __restrict__ gm_workspace_;

    int64_t batchSize_;
    int64_t fftN_;
    int32_t radixListLen_;
    int32_t isInverse_;
    int64_t workspaceOffsets_[5];
    
    __gm__ int32_t * __restrict__ gm_radix_arr_;
    __gm__ int64_t * __restrict__ gm_M_arr_;
    __gm__ int64_t * __restrict__ gm_dft_offset_arr_;
    __gm__ int64_t * __restrict__ gm_tw_offset_arr_;
    __gm__ int64_t * __restrict__ gm_currentBatch_arr_;
};

__aicore__ inline void FftR2CKernel::Init(
    __gm__ float * __restrict__ gm_input,
    __gm__ float * __restrict__ gm_dft_matrix_array,
    __gm__ float * __restrict__ gm_tw_matrix_array,
    __gm__ float * __restrict__ gm_tw_post_process,
    __gm__ float * __restrict__ radix_list,
    __gm__ float * __restrict__ gm_output,
    __gm__ float * __restrict__ gm_workspace,
    __gm__ uint8_t * __restrict__ gm_tiling_para)
{
    gm_input_ = gm_input;
    gm_dft_matrix_array_ = gm_dft_matrix_array;
    gm_tw_matrix_array_ = gm_tw_matrix_array;
    gm_tw_post_process_ = gm_tw_post_process;
    gm_output_ = gm_output;
    gm_workspace_ = gm_workspace;

    auto tiling_buf = reinterpret_cast<__gm__ uint8_t *>(gm_tiling_para);

    batchSize_ = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf));
    fftN_ = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 8));
    radixListLen_ = (*(__gm__ int32_t *)((__gm__ uint8_t *)tiling_buf + 16));
    isInverse_ = (*(__gm__ int32_t *)((__gm__ uint8_t *)tiling_buf + 20));

    for (int32_t i = 0; i < 5; i++) {
        workspaceOffsets_[i] = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 24 + 8 * i));
    }
    
    int64_t baseOffset = 24 + 5 * 8;
    gm_radix_arr_ = (__gm__ int32_t *)((__gm__ uint8_t *)tiling_buf + baseOffset);
    gm_M_arr_ = (__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + baseOffset + MAX_FFT_STAGES * 4);
    gm_dft_offset_arr_ = (__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + baseOffset + MAX_FFT_STAGES * 4 + MAX_FFT_STAGES * 8);
    gm_tw_offset_arr_ = (__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + baseOffset + MAX_FFT_STAGES * 4 + MAX_FFT_STAGES * 8 * 2);
    gm_currentBatch_arr_ = (__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + baseOffset + MAX_FFT_STAGES * 4 + MAX_FFT_STAGES * 8 * 3);
}

__aicore__ inline void FftR2CKernel::Process()
{
    Simt::VF_CALL<FftR2CPackRealToComplex>(
        Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1},
        gm_input_,
        gm_workspace_,
        batchSize_,
        fftN_,
        workspaceOffsets_[0]);
    
    int64_t tempBatch = batchSize_;
    for (int32_t step = 0; step < radixListLen_; step++) {
        __gm__ float * __restrict__ inputBuf;
        __gm__ float * __restrict__ outputBuf;
        if (step % 2 == 0) {
            inputBuf  = (__gm__ float *)((__gm__ uint8_t *)gm_workspace_ + workspaceOffsets_[0]);
            outputBuf = (__gm__ float *)((__gm__ uint8_t *)gm_workspace_ + workspaceOffsets_[1]);
        } else {
            inputBuf  = (__gm__ float *)((__gm__ uint8_t *)gm_workspace_ + workspaceOffsets_[1]);
            outputBuf = (__gm__ float *)((__gm__ uint8_t *)gm_workspace_ + workspaceOffsets_[0]);
        }

        __gm__ float * __restrict__ dftMat = gm_dft_matrix_array_ + gm_dft_offset_arr_[step];
        __gm__ float * __restrict__ twMat  = gm_tw_matrix_array_  + gm_tw_offset_arr_[step];

        Simt::VF_CALL<FftR2CStockhamForwardStage>(
            Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1},
            inputBuf,
            outputBuf,
            dftMat,
            twMat,
            gm_radix_arr_[step],
            gm_M_arr_[step],
            tempBatch,
            step,
            radixListLen_);

        tempBatch *= gm_radix_arr_[step];
    }

    bool useWs1 = (radixListLen_ % 2 == 1);
    tempBatch = batchSize_;
    
    for (int32_t s = 0; s < radixListLen_; s++) {
        tempBatch *= gm_radix_arr_[s];
    }
    
    for (int32_t step = radixListLen_ - 1; step >= 0; step--) {
        __gm__ float * __restrict__ inputBuf;
        __gm__ float * __restrict__ outputBuf;
        if (useWs1) {
            inputBuf  = (__gm__ float *)((__gm__ uint8_t *)gm_workspace_ + workspaceOffsets_[1]);
            outputBuf = (__gm__ float *)((__gm__ uint8_t *)gm_workspace_ + workspaceOffsets_[0]);
        } else {
            inputBuf  = (__gm__ float *)((__gm__ uint8_t *)gm_workspace_ + workspaceOffsets_[0]);
            outputBuf = (__gm__ float *)((__gm__ uint8_t *)gm_workspace_ + workspaceOffsets_[1]);
        }
        
        Simt::VF_CALL<FftR2CTransposeStage>(
            Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1},
            inputBuf,
            outputBuf,
            gm_radix_arr_[step],
            gm_M_arr_[step],
            tempBatch / gm_radix_arr_[step]);
        
        tempBatch /= gm_radix_arr_[step];
        useWs1 = !useWs1;
    }

    Simt::VF_CALL<FftR2CPostProcess>(
        Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1},
        gm_workspace_,
        gm_tw_post_process_,
        gm_output_,
        batchSize_,
        fftN_,
        workspaceOffsets_[0]);
    
    return;
}

extern "C" __global__ __aicore__ void fft_r2c(
    __gm__ float * __restrict__ gm_input,
    __gm__ float * __restrict__ gm_dft_matrix_array,
    __gm__ float * __restrict__ gm_tw_matrix_array,
    __gm__ float * __restrict__ gm_tw_post_process,
    __gm__ float * __restrict__ radix_list,
    __gm__ float * __restrict__ gm_output,
    __gm__ float * __restrict__ gm_workspace,
    __gm__ uint8_t * __restrict__ gm_tiling_para)
{
    FftR2CKernel kernel;
    kernel.Init(gm_input, gm_dft_matrix_array, gm_tw_matrix_array, gm_tw_post_process, radix_list,
                gm_output, gm_workspace, gm_tiling_para);
    kernel.Process();
}