# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(dot_src
    ${CMAKE_CURRENT_LIST_DIR}/dot_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cdot/cdot_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cdot/tiling/cdot_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sdot/sdot_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sdot/tiling/sdot_tiling.cpp
)

add_operation(DotOperation "${dot_src}")

add_kernel(cdot ascend910b mix 
cdot/kernel/cdot.cce
    CdotC64Kernel)

add_kernel(sdot ascend910b mix 
sdot/kernel/sdot.cce
    SdotF32Kernel)
