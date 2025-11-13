# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(cgemv_batched_src
    ${CMAKE_CURRENT_LIST_DIR}/cgemv_batched_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cgemv_batched/cgemv_batched_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cgemv_batched/tiling/cgemv_batched_tiling.cpp
)

add_operation(CgemvBatchedOperation "${cgemv_batched_src}")

add_kernel(cgemv_batched ascend910b vector
cgemv_batched/kernel/cgemv_batched.cpp
    CgemvBatchedC32C64Kernel)