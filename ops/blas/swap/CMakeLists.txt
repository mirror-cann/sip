# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(swap_src
    ${CMAKE_CURRENT_LIST_DIR}/swap_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sswap/sswap_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sswap/tiling/sswap_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cswap/cswap_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cswap/tiling/cswap_tiling.cpp
)

add_operation(SwapOperation "${swap_src}")

add_kernel(sswap ascend910b mix 
sswap/kernel/sswap.cce
    SswapF32Kernel)

add_kernel(cswap ascend910b mix 
cswap/kernel/cswap.cce
    CswapC64Kernel)