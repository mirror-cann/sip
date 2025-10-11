# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(hcgemm_batched_src
    ${CMAKE_CURRENT_LIST_DIR}/hcgemm_batched_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/hcgemm_batched/hcgemm_batched_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/hcgemm_batched/tiling/hcgemm_batched_tiling.cpp
)

add_operation(HCgemmBatchedOperation "${hcgemm_batched_src}")

add_kernel(hcgemm_batched ascend910b mix 
    hcgemm_batched/kernel/hcgemm_batched.cpp
    HCgemmBatchedC32Kernel)
