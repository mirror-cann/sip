# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(dft_src
    ${CMAKE_CURRENT_LIST_DIR}/dft_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dft/dft_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dft/tiling/dft_tiling.cpp
)
 
add_operation(DftOperation "${dft_src}")

add_kernel(dft ascend910b cube
    dft/kernel/dft.cce
    DftC64Kernel)
