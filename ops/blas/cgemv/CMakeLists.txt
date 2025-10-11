# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(cgemv_src
    ${CMAKE_CURRENT_LIST_DIR}/cgemv_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cgemv/cgemv_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cgemv/tiling/cgemv_tiling.cpp
)

add_operation(CgemvOperation "${cgemv_src}")

add_kernel(cgemv_do_trans ascend910b mix
cgemv/kernel/cgemv_do_trans.cce
    CgemvDoTransC64Kernel)

add_kernel(cgemv_no_trans ascend910b mix
cgemv/kernel/cgemv_no_trans.cce
    CgemvNoTransC64Kernel)

