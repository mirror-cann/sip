# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(cgerc_src
    ${CMAKE_CURRENT_LIST_DIR}/cgerc_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cgerc/cgerc_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cgerc/tiling/cgerc_tiling.cpp
)

add_operation(CgercOperation "${cgerc_src}")

add_kernel(cgerc ascend910b mix 
    cgerc/kernel/cgerc.cce
    CgercC64Kernel)
