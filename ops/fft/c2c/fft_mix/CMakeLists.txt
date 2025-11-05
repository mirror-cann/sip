# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(fft_mix_src
    ${CMAKE_CURRENT_LIST_DIR}/fft_mix_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fft_mix/fft_mix_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fft_mix/tiling/fft_mix_tiling.cpp
)

add_operation(FftMixOperation "${fft_mix_src}")

add_kernel(fft_mix_1 ascend910b mix
    fft_mix/kernel/fft_mix_1.cce
    FftMix1C64Kernel)

add_kernel(fft_mix_2 ascend910b mix
    fft_mix/kernel/fft_mix_2.cce
    FftMix2C64Kernel)

add_kernel(fft_mix_3 ascend910b mix
    fft_mix/kernel/fft_mix_3.cce
    FftMix3C64Kernel)

add_kernel(fft_mix_4 ascend910b mix
    fft_mix/kernel/fft_mix_4.cce
    FftMix4C64Kernel)

add_kernel(fft_mix_5 ascend910b mix
    fft_mix/kernel/fft_mix_5.cce
    FftMix5C64Kernel)

add_kernel(fft_mix_6 ascend910b mix
    fft_mix/kernel/fft_mix_6.cce
    FftMix6C64Kernel)
