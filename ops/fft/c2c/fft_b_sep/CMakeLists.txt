# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(fft_b_sep_src
    ${CMAKE_CURRENT_LIST_DIR}/fft_b_sep_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fft_b_sep/fft_b_sep_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fft_b_sep/tiling/fft_b_sep_tiling.cpp
)

set(CATLASS_INCLUDES 
    ${OPS_THIRD_PARTY_DIR}/catlass/include
    $ENV{ASCEND_HOME_PATH}/include
    ${PROJECT_SOURCE_DIR}/ops/include/template
)

add_operation(FftBSepOperation "${fft_b_sep_src}")

add_kernel(fft_b_sep ascend910b mix
    fft_b_sep/kernel/fft_b_sep.cpp
    FftBSepC64Kernel
    INCLUDE_DIRECTORIES ${CATLASS_INCLUDES})
