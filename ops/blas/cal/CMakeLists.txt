# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(cal_src
    ${CMAKE_CURRENT_LIST_DIR}/cal_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cal/cal_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cal/tiling/cal_tiling.cpp
)

add_operation(CalOperation "${cal_src}")

# target_link_libraries(CalOperation PRIVATE ops_common_tiling)

add_kernel(sscal ascend910b vector
    cal/kernel/sscal/sscal.cpp
    SscalF32Kernel)

add_kernel(cscal ascend910b mix
    cal/kernel/cscal/cscal.cce
    CscalC64Kernel)
