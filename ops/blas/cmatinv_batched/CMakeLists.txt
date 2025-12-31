# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(cmatinv_batched_src
    ${CMAKE_CURRENT_LIST_DIR}/cmatinv_batched_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cmatinv_batched/cmatinv_batched_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cmatinv_batched/tiling/cmatinv_batched_tiling.cpp
)

add_operation(CmatinvBatchedOperation "${cmatinv_batched_src}")

add_kernel(cmatinv_batched ascend910b vector 
    cmatinv_batched/kernel/cmatinv_batched.cpp
    CmatinvBatchedC64Kernel)