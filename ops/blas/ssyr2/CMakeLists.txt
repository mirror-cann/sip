# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(ssyr2_src
    ${CMAKE_CURRENT_LIST_DIR}/ssyr2_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ssyr2/ssyr2_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ssyr2/tiling/ssyr2_tiling.cpp
)

add_operation(Ssyr2Operation "${ssyr2_src}")

add_kernel(ssyr2 ascend910b mix 
ssyr2/kernel/ssyr2.cce
    Ssyr2F32Kernel)

