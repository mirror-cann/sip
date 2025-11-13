# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(nrm2_src
    ${CMAKE_CURRENT_LIST_DIR}/nrm2_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/nrm2/nrm2_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/nrm2/tiling/nrm2_tiling.cpp
)

add_operation(Nrm2Operation "${nrm2_src}")

# target_link_libraries(Nrm2Operation PRIVATE ops_common_tiling)

add_kernel(nrm2 ascend910b mix
    nrm2/kernel/nrm2.cpp
    Nrm2F32Kernel)
