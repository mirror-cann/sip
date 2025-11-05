# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(copy_src
    ${CMAKE_CURRENT_LIST_DIR}/copy_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/copy/copy_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/copy/tiling/copy_tiling.cpp
)
 
add_operation(CopyOperation "${copy_src}")
 
# target_link_libraries(CopyOperation PRIVATE ops_common_tiling)
 
add_kernel(copy ascend910b vector
    copy/kernel/copy.cpp
    CopyF32Kernel)