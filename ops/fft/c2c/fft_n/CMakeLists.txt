# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(fft_n_src
    ${CMAKE_CURRENT_LIST_DIR}/fft_n_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fft_n/fft_n_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fft_n/tiling/fft_n_tiling.cpp
)

add_operation(FftNOperation "${fft_n_src}")

add_kernel(fft_n ascend910b mix
    fft_n/kernel/fft_n.cce
    FftNC64Kernel)
