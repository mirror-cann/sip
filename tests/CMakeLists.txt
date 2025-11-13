# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 设置链接库地址
set(LINKED_LIBRARIES blas base fft filter interpolation utils asdsip_core dl c_sec mmpa)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)

# 包括PyTorch头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/core/include)
include_directories(${PROJECT_SOURCE_DIR}/core/utils/include)

if("$ENV{TEST_TYPE}" STREQUAL "UT")
    add_subdirectory(ut)
endif()
if("$ENV{TEST_TYPE}" STREQUAL "ST")
    add_subdirectory(st)
endif()
if("$ENV{TEST_TYPE}" STREQUAL "FT")
    add_subdirectory(fuzz)
endif()
