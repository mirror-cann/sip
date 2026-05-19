# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(mul_src
    ${CMAKE_CURRENT_LIST_DIR}/mul_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mul/mul_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mul/tiling/mul_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mul/tiling/arch35/mul_tiling_arch35.cpp
)

add_operation(MulOperation "${mul_src}")

add_kernel(mul_c64 ascend910b vector
    mul/op_kernel/mul_c64.cpp
    MulC64Kernel)

add_kernel(mul_c32 ascend910b vector
    mul/op_kernel/mul_c32.cpp
    MulC32Kernel)

add_kernel(mul_arch35 ascend950 vector
    mul/op_kernel/arch35/mul_arch35.cpp
    MulArch35Kernel)
