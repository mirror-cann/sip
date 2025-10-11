# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 加到lib中
add_library(asdsip SHARED asdsip_api_binder.cpp)

# 链接依赖库到你的可执行文件
target_link_libraries(asdsip ${LINKED_LIBRARIES})

target_compile_options(asdsip PRIVATE
    -Wno-unused-function
    -Wno-attributes
    -Wno-maybe-uninitialized
)

install(TARGETS asdsip DESTINATION st)