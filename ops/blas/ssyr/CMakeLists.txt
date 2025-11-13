# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(ssyr_src
    ${CMAKE_CURRENT_LIST_DIR}/ssyr_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ssyr/ssyr_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ssyr/tiling/ssyr_tiling.cpp
)

add_operation(SsyrOperation "${ssyr_src}")

add_kernel(ssyr ascend910b vector 
    ssyr/kernel/ssyr.cce
    SsyrF32Kernel)

