# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(strmv_srcs
    ${CMAKE_CURRENT_LIST_DIR}/strmv/strmv_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/strmv/tiling/strmv_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/strmv_operation.cpp
)

add_operation(StrmvOperation "${strmv_srcs}")

add_kernel(strmv ascend910b mix
    strmv/kernel/strmv.cce
    StrmvF32Kernel)