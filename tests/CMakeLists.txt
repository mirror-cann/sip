# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 设置torch lib地址
set(PYTORCH_LIBRARY_DIRS $ENV{PYTHON_LIB_PATH}/torch/lib/  $ENV{PYTHON_LIB_PATH}/torch.libs/ $ENV{PYTHON_LIB_PATH}/torch_npu/lib/)

# 设置链接库地址
set(LINKED_LIBRARIES torch torch_python torch_npu blas base fft filter interpolation utils asdsip_core dl c_sec mmpa)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)

# 包括PyTorch头文件目录
include_directories($ENV{PYTHON_INCLUDE_PATH}/)
include_directories($ENV{PYTHON_LIB_PATH}/torch/include/)
include_directories($ENV{PYTHON_LIB_PATH}/torch/include/torch/csrc/api/include/)
include_directories($ENV{PYTHON_LIB_PATH}/torch_npu/include/)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/core/include)
include_directories(${PROJECT_SOURCE_DIR}/core/utils/include)
# 链接PyTorch库
link_directories(${PYTORCH_LIBRARY_DIRS})

if("$ENV{TEST_TYPE}" STREQUAL "UT")
    add_subdirectory(ut)
endif()
if("$ENV{TEST_TYPE}" STREQUAL "ST")
    add_subdirectory(st)
endif()
if("$ENV{TEST_TYPE}" STREQUAL "FT")
    add_subdirectory(fuzz)
endif()