# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
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
