# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(fft_c2c_arch35_src
    ${CMAKE_CURRENT_LIST_DIR}/fft_c2c_arch35_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fft_c2c_arch35/fft_c2c_arch35_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fft_c2c_arch35/tiling/fft_c2c_arch35_tiling.cpp
)

set(CATLASS_INCLUDES
    ${OPS_THIRD_PARTY_DIR}/catlass/include
    $ENV{ASCEND_HOME_PATH}/include
    ${PROJECT_SOURCE_DIR}/ops/include/template
)

add_operation(FftC2CArch35Operation "${fft_c2c_arch35_src}")

add_kernel(fft_c2c_arch35_multi_core ascend950 vector
    fft_c2c_arch35/kernel/arch35/fft_c2c_multi_core.cpp
    FftC2CArch35C64Kernel
    INCLUDE_DIRECTORIES ${CATLASS_INCLUDES})
