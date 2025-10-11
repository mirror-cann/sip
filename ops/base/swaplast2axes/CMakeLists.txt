# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(swap_last_2axes_src
    ${CMAKE_CURRENT_LIST_DIR}/swap_last_2axes_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/swap_last_2axes_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/swap_last_2axes_tiling.cpp
)
 
add_operation(SwapLast2AxesOperation "${swap_last_2axes_src}")

add_kernel(swapLast2AxesComplex ascend910b vector
    swaplast2axes_complex/kernel/swapLast2Axes.cce
    SwapLast2AxesC64Kernel)
