# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

file(GLOB_RECURSE TEST_UTIL_SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/test_util/*.cpp")
add_library(asdsip_test_util ${TEST_UTIL_SOURCE_FILES})
target_include_directories(asdsip_test_util PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(asdsip_test_util PUBLIC asdsip asdsip_core torch torch_cpu c10)