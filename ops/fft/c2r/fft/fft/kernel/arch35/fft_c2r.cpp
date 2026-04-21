/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define CATLASS_ARCH 3510

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

constexpr uint32_t VF_MAX_THREAD_NUM = 256;
// Maximum FFT stages: for fftN up to 2^32, but practical limit is much smaller
// For fftN=8192: log2(8192)=13, for fftN=262144: log2=18
// Using 20 is sufficient and saves UB space
constexpr int32_t MAX_FFT_STAGES = 32;

// ============================================================================
// Callee helper functions
// ============================================================================

/**
 * Complex multiplication: (ar + ai*j) * (br + bi*j) = (cr + ci*j)
 */
__simt_callee__ inline void ComplexMul(
    float ar, float ai, float br, float bi, float &cr, float &ci)
{
    cr = ar * br - ai * bi;
    ci = ar * bi + ai * br;
}

/**
 * Find the smallest radix from {2,3,5,7} that divides n.
 * Matches Host-side InitRadix decomposition order.
 */
__simt_callee__ inline int32_t FindRadix(int64_t n)
{
    if (n % 2 == 0) return 2;
    if (n % 3 == 0) return 3;
    if (n % 5 == 0) return 5;
    if (n % 7 == 0) return 7;
    return 0;
}

// ============================================================================
// Phase 1: Hermitian Symmetric Expansion
// ============================================================================
/**
 * Expand N/2+1 complex inputs to N complex values using Hermitian symmetry.
 *
 * Corresponds to Python:
 *   X_full = np.zeros((batch, target_N), dtype=np.complex128)
 *   X_full[:, :D] = X_cplx
 *   for i in range(1, (target_N + 1) // 2):
 *       X_full[:, target_N - i] = np.conj(X_cplx[:, i])
 *
 * Input:  gm_input [batch, D] complex64, D = fftN/2+1, float interleaved
 * Output: workspace [batch, fftN] complex64, float interleaved
 */
__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) __aicore__ void FftC2RHermitianExpand(
    __gm__ float * __restrict__ gm_input,
    __gm__ float * __restrict__ gm_workspace,
    int64_t batchSize,
    int64_t fftN,
    int64_t wsOffset)
{
    int64_t D = fftN / 2 + 1;  // Number of input complex points (N/2+1)
    int64_t totalComplex = batchSize * fftN;

    int64_t tid = static_cast<int64_t>(Simt::GetThreadIdx<0>()) +
                  static_cast<int64_t>(Simt::GetBlockIdx()) * static_cast<int64_t>(Simt::GetThreadNum<0>());
    int64_t stride = static_cast<int64_t>(Simt::GetBlockNum()) * static_cast<int64_t>(Simt::GetThreadNum<0>());

    // Convert wsOffset from bytes to float element index
    int64_t wsFloatOffset = wsOffset / sizeof(float);

    for (int64_t idx = tid; idx < totalComplex; idx += stride) {
        int64_t b = idx / fftN;
        int64_t k = idx % fftN;

        int64_t outFloatIdx = wsFloatOffset + (b * fftN + k) * 2;

        if (k < D) {
            // Direct copy: G[k] = F[k] for k = 0..D-1 (includes DC and Nyquist)
            int64_t inFloatIdx = (b * D + k) * 2;
            gm_workspace[outFloatIdx]     = gm_input[inFloatIdx];
            gm_workspace[outFloatIdx + 1] = gm_input[inFloatIdx + 1];
        } else {
            // Conjugate symmetric: G[k] = conj(F[N-k]) for k = D..N-1
            // i = N - k is the index in input array
            int64_t i = fftN - k;
            // i must be in range [1, D-1] for conjugate symmetry
            // (i=0 handled by k<D branch, Nyquist point handled separately for even N)
            if (i >= 1 && i < D) {
                int64_t inFloatIdx = (b * D + i) * 2;
                gm_workspace[outFloatIdx]     =  gm_input[inFloatIdx];      // real(F[i])
                gm_workspace[outFloatIdx + 1] = -gm_input[inFloatIdx + 1];  // -imag(F[i])
            } else {
                // This should not happen for valid N, but zero-fill for safety
                gm_workspace[outFloatIdx]     = 0.0f;
                gm_workspace[outFloatIdx + 1] = 0.0f;
            }
        }
    }
}

// ============================================================================
// DEBUG: Copy workspace real parts to output for verification
// ============================================================================
__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) __aicore__ void FftC2RDebugCopyWorkspaceToOutput(
    __gm__ float * __restrict__ gm_workspace,
    __gm__ float * __restrict__ gm_output,
    int64_t batchSize,
    int64_t fftN,
    int64_t wsOffset)
{
    int64_t totalFloats = batchSize * fftN;  // Only copy real parts (batch * N floats)

    int64_t tid = static_cast<int64_t>(Simt::GetThreadIdx<0>()) +
                  static_cast<int64_t>(Simt::GetBlockIdx()) * static_cast<int64_t>(Simt::GetThreadNum<0>());
    int64_t stride = static_cast<int64_t>(Simt::GetBlockNum()) * static_cast<int64_t>(Simt::GetThreadNum<0>());

    // wsOffset is in bytes, convert to float index
    // Workspace stores complex64: [re0, im0, re1, im1, ...]
    // We only copy real parts (even indices)
    int64_t wsFloatOffset = wsOffset / sizeof(float);

    for (int64_t idx = tid; idx < totalFloats; idx += stride) {
        gm_output[idx] = gm_workspace[wsFloatOffset + idx];
    }
}


// ============================================================================
// Phase 2a: Stockham Forward Pass - Single Stage VF Function
// ============================================================================
/**
 * @brief Execute ONE stage of Stockham FFT forward pass (DFT + Twiddle)
 * 
 * Algorithm:
 *   For each output element outIdx = (batchIdx * radix + u) * M + n2:
 *     1. DFT: result[u,n2] = sum_v DFT[u,v] * input[v,n2]  for v in [0, radix-1]
 *     2. Twiddle: result[u,n2] *= TW[u,n2]  (skip for last stage)
 * 
 * Python reference: debug_stage0_kernel_logic.py
 *   DFT matrix format (interleaved rows):
 *     [re_row0(n cols), im_row0(n cols), re_row1(n cols), im_row1(n cols), ...]
 *   Twiddle matrix: same format as DFT
 *   
 * Index mapping:
 *   outIdx -> (batchIdx, u, n2)
 *     batchIdx = outIdx / (radix * M)
 *     rem      = outIdx % (radix * M)
 *     u        = rem / M
 *     n2       = rem % M
 *   
 *   Input index for v: (batchIdx * radix * M + v * M + n2) * 2
 *   Output index:      (batchIdx * radix + u) * M + n2) * 2
 * 
 * @param inputBuf      [IN]  Input buffer [batch*radix*M] complex
 * @param outputBuf     [OUT] Output buffer [batch*radix*M] complex
 * @param dftMat        [IN]  DFT matrix [radix*radix] complex (interleaved format)
 * @param twMat         [IN]  Twiddle matrix [radix*M] complex (interleaved format)
 * @param radix         [IN]  Radix for this stage (2, 3, 5, or 7)
 * @param M             [IN]  M = N / product(radix_0...radix_s)
 * @param currentBatch  [IN]  Current batch size
 * @param step          [IN]  Current stage index (0-based)
 * @param radixListLen  [IN]  Total number of stages
 */
__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) __aicore__ void FftC2RStockhamForwardStage(
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

        // DFT matrix multiply - FULLY UNROLLED for radix {2,3,5,7}
        float resultRe = 0.0f;
        float resultIm = 0.0f;

        // Helper macro for one DFT term
        // Matrix format: [re_row0, im_row0, re_row1, im_row1, ...]
        // Each row has 'radix' elements
        // Access (u,v): re = dftMat[(2*u)*radix + v], im = dftMat[(2*u+1)*radix + v]
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

        // Fully unrolled DFT computation
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

        // Twiddle factor multiply (skip for last stage)
        if (step < radixListLen - 1) {
            // Twiddle matrix format: [re_row0, im_row0, re_row1, im_row1, ...]
            // Access (k1, n2): re = twMat[(2*k1)*M + n2], im = twMat[(2*k1+1)*M + n2]
            float tRe = twMat[(2 * u) * M + n2];
            // float tRe = 1.0f;
            float tIm = twMat[(2 * u + 1) * M + n2];
            // float tIm = 0.0f;

            float tmpRe, tmpIm;
            ComplexMul(resultRe, resultIm, tRe, tIm, tmpRe, tmpIm);
            resultRe = tmpRe;
            resultIm = tmpIm;

            // resultRe = tRe;
            // resultIm = tIm;
        }

        // Write to output buffer
        int64_t outComplexIdx = (batchIdx * radix + u) * M + n2;
        int64_t outFloatIdx = outComplexIdx * 2;
        outputBuf[outFloatIdx]     = resultRe;
        outputBuf[outFloatIdx + 1] = resultIm;
    }
}

// ============================================================================
// Phase 2: Stockham Backward Pass - Transpose interleave
// ============================================================================
/**
 * Execute Stockham FFT backward pass iteratively (reverse stage order):
 *   For each stage step = radixListLen-1..0:
 *     1. View data as [currentBatch, radix, M]
 *     2. Transpose to [currentBatch, M, radix]
 *     3. Reshape to [currentBatch, radix*M]  (contiguous output)
 *
 * This corresponds to the recursive return in Python _execute_fft_plan:
 *   X_restored = X_recursive.reshape(current_batch, radix, M)
 *   X_out = np.transpose(X_restored, axes=(0, 2, 1)).reshape(current_batch, radix * M)
 *
 * Buffer convention:
 *   The first transpose reads from the forward pass output buffer.
 *   Forward pass output: ws1 if radixListLen is odd, ws0 if even.
 *   We track buffer with forwardEndBuf (0=ws0, 1=ws1).
 *   Each transpose step swaps the buffer.
 *   After radixListLen transpose steps, final buffer parity =
 *     (forwardEndBuf + radixListLen) % 2
 */
// ============================================================================
// Phase 2b: Stockham Backward Pass - Single Transpose Stage VF Function
// ============================================================================
// Process SINGLE transpose stage only (no loop), called from __aicore__ Process()
__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) __aicore__ void FftC2RTransposeStage(
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

// ============================================================================
// Phase 3: Take real part and normalize
// ============================================================================
/**
 * Take real part of C2C IFFT result and normalize by fftN.
 * Input:  workspace [batch, fftN] complex64 (float interleaved)
 * Output: gm_output [batch, fftN] float
 */
__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) __aicore__ void FftC2RRealNormalize(
    __gm__ float * __restrict__ gm_workspace,
    __gm__ float * __restrict__ gm_output,
    int64_t batchSize,
    int64_t fftN,
    int64_t wsOffset)
{
    int64_t totalElements = batchSize * fftN;
    float invN = 1.0f / static_cast<float>(fftN);

    int64_t tid = static_cast<int64_t>(Simt::GetThreadIdx<0>()) +
                  static_cast<int64_t>(Simt::GetBlockIdx()) * static_cast<int64_t>(Simt::GetThreadNum<0>());
    int64_t stride = static_cast<int64_t>(Simt::GetBlockNum()) * static_cast<int64_t>(Simt::GetThreadNum<0>());

    // Convert wsOffset from bytes to float element index
    int64_t wsFloatOffset = wsOffset / sizeof(float);

    for (int64_t idx = tid; idx < totalElements; idx += stride) {
        float re = gm_workspace[wsFloatOffset + idx * 2];
        gm_output[idx] = re * invN;
    }
}

// ============================================================================
// FftC2RKernel class
// ============================================================================
class FftC2RKernel {
public:
    __aicore__ inline FftC2RKernel() {}

    __aicore__ inline void Init(
        __gm__ float * __restrict__ gm_input,
        __gm__ float * __restrict__ gm_dft_matrix_array,
        __gm__ float * __restrict__ gm_tw_matrix_array,
        __gm__ float * __restrict__ radix_list,
        __gm__ float * __restrict__ gm_output,
        __gm__ float * __restrict__ gm_workspace,
        __gm__ uint8_t * __restrict__ gm_tiling_para);

    __aicore__ inline void Process();

private:
    __gm__ float * __restrict__ gm_input_;
    __gm__ float * __restrict__ gm_dft_matrix_array_;
    __gm__ float * __restrict__ gm_tw_matrix_array_;
    __gm__ float * __restrict__ gm_output_;
    __gm__ float * __restrict__ gm_workspace_;

    int64_t batchSize_;
    int64_t fftN_;
    int32_t radixListLen_;
    int32_t isInverse_;
    int64_t workspaceOffsets_[5];
    
    // Pointers to precomputed plan data in tiling (no local arrays to avoid VLOOP)
    __gm__ int32_t * __restrict__ gm_radix_arr_;
    __gm__ int64_t * __restrict__ gm_M_arr_;
    __gm__ int64_t * __restrict__ gm_dft_offset_arr_;
    __gm__ int64_t * __restrict__ gm_tw_offset_arr_;
    __gm__ int64_t * __restrict__ gm_currentBatch_arr_;
};

__aicore__ inline void FftC2RKernel::Init(
    __gm__ float * __restrict__ gm_input,
    __gm__ float * __restrict__ gm_dft_matrix_array,
    __gm__ float * __restrict__ gm_tw_matrix_array,
    __gm__ float * __restrict__ radix_list,
    __gm__ float * __restrict__ gm_output,
    __gm__ float * __restrict__ gm_workspace,
    __gm__ uint8_t * __restrict__ gm_tiling_para)
{
    gm_input_ = gm_input;
    gm_dft_matrix_array_ = gm_dft_matrix_array;
    gm_tw_matrix_array_ = gm_tw_matrix_array;
    gm_output_ = gm_output;
    gm_workspace_ = gm_workspace;

    // get tiling args (all precomputed on host side)
    auto tiling_buf = reinterpret_cast<__gm__ uint8_t *>(gm_tiling_para);

    batchSize_ = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf));
    fftN_ = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 8));
    radixListLen_ = (*(__gm__ int32_t *)((__gm__ uint8_t *)tiling_buf + 16));
    isInverse_ = (*(__gm__ int32_t *)((__gm__ uint8_t *)tiling_buf + 20));

    for (int32_t i = 0; i < 5; i++) {
        workspaceOffsets_[i] = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 24 + 8 * i));
    }
    
    // Get pointers to precomputed arrays in tiling data
    int64_t baseOffset = 24 + 5 * 8;  // After workspaceOffsets
    gm_radix_arr_ = (__gm__ int32_t *)((__gm__ uint8_t *)tiling_buf + baseOffset);
    gm_M_arr_ = (__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + baseOffset + MAX_FFT_STAGES * 4);
    gm_dft_offset_arr_ = (__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + baseOffset + MAX_FFT_STAGES * 4 + MAX_FFT_STAGES * 8);
    gm_tw_offset_arr_ = (__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + baseOffset + MAX_FFT_STAGES * 4 + MAX_FFT_STAGES * 8 * 2);
    gm_currentBatch_arr_ = (__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + baseOffset + MAX_FFT_STAGES * 4 + MAX_FFT_STAGES * 8 * 3);
}

__aicore__ inline void FftC2RKernel::Process()
{
    // Phase 1: Hermitian symmetric expansion
    // Input:  gm_input_ [batch, N/2+1] complex
    // Output: workspace[0] [batch, N] complex (hermitian symmetric)
    // 
    // Python reference:
    //   x[n] = IFFT(X[:N/2+1])  # Hermitian symmetric expansion
    //   x_full[k] = X[k] for k <= N/2
    //   x_full[k] = conj(X[N-k]) for k > N/2
    Simt::VF_CALL<FftC2RHermitianExpand>(
        Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1},
        gm_input_,
        gm_workspace_,
        batchSize_,
        fftN_,
        workspaceOffsets_[0]);
    
    // Phase 2a: Stockham Forward Pass (DFT + Twiddle multiplication)
    // Input:  workspace[0] [batch, N] complex (hermitian symmetric)
    // Output: workspace[1] or workspace[0] (ping-pong) [batch*N] complex
    // 
    // Algorithm: Multi-stage Stockham FFT
    //   For each stage s (radix r_s, M_s = N / product(r_0...r_s)):
    //     1. DFT matrix multiply: X_out[u, n2] = sum_v DFT[u,v] * X_in[v, n2]
    //     2. Twiddle factor: X_out[u, n2] *= TW[u, n2]  (skip for last stage)
    int64_t tempBatch = batchSize_;
    for (int32_t step = 0; step < radixListLen_; step++) {
    // for (int32_t step = 0; step < 1; step++) {
        __gm__ float * __restrict__ inputBuf;
        __gm__ float * __restrict__ outputBuf;
        if (step % 2 == 0) {
            inputBuf  = (__gm__ float *)((__gm__ uint8_t *)gm_workspace_ + workspaceOffsets_[0]);
            outputBuf = (__gm__ float *)((__gm__ uint8_t *)gm_workspace_ + workspaceOffsets_[1]);
        } else {
            inputBuf  = (__gm__ float *)((__gm__ uint8_t *)gm_workspace_ + workspaceOffsets_[1]);
            outputBuf = (__gm__ float *)((__gm__ uint8_t *)gm_workspace_ + workspaceOffsets_[0]);
        }


        // Get DFT/TW matrix pointers for this stage
        __gm__ float * __restrict__ dftMat = gm_dft_matrix_array_ + gm_dft_offset_arr_[step];
        __gm__ float * __restrict__ twMat  = gm_tw_matrix_array_  + gm_tw_offset_arr_[step];


        // Call single-stage VF function
        Simt::VF_CALL<FftC2RStockhamForwardStage>(
            Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1},
            inputBuf,
            outputBuf,
            dftMat,
            twMat,
            gm_radix_arr_[step], // radix
            gm_M_arr_[step], // M
            tempBatch, // currentBatch
            step,
            radixListLen_);

        // Update tempBatch for next stage
        tempBatch *= gm_radix_arr_[step];
    }

    // Phase 2b: Stockham Backward Pass (transpose interleave)
    // Forward pass output: ws1 if radixListLen is odd, ws0 if even
    bool useWs1 = (radixListLen_ % 2 == 1);
    tempBatch = batchSize_;
    
    // Compute tempBatch at the END of forward pass
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
        
        Simt::VF_CALL<FftC2RTransposeStage>(
            Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1},
            inputBuf,
            outputBuf,
            gm_radix_arr_[step],
            gm_M_arr_[step],
            tempBatch / gm_radix_arr_[step]);
        
        tempBatch /= gm_radix_arr_[step];
        useWs1 = !useWs1;
    }

    // Phase 3: Take real part and normalize
    // Final Stockham result is in ws0 (workspaceOffsets[0])
    Simt::VF_CALL<FftC2RRealNormalize>(
        Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1},
        gm_workspace_,
        gm_output_,
        batchSize_,
        fftN_,
        workspaceOffsets_[0]);
    
    return;
}

// ============================================================================
// Kernel entry point (DO NOT MODIFY)
// ============================================================================
/**
 * FFT C2R kernel for Ascend 950 (arch35)
 */
extern "C" __global__ __aicore__ void fft_c2r(
    __gm__ float * __restrict__ gm_input,
    __gm__ float * __restrict__ gm_dft_matrix_array,
    __gm__ float * __restrict__ gm_tw_matrix_array,
    __gm__ float * __restrict__ radix_list,
    __gm__ float * __restrict__ gm_output,
    __gm__ float * __restrict__ gm_workspace,
    __gm__ uint8_t * __restrict__ gm_tiling_para)
{
    FftC2RKernel kernel;
    kernel.Init(gm_input, gm_dft_matrix_array, gm_tw_matrix_array, radix_list,
                gm_output, gm_workspace, gm_tiling_para);
    kernel.Process();
}
