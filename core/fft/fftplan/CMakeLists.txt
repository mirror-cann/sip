# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_library(fft_plan STATIC fft_plan.cpp fft_plan_cache.cpp)

target_include_directories(fft_plan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# target_link_libraries(fft_plan PUBLIC base)
target_link_libraries(fft_plan PUBLIC fft_utils)
target_link_libraries(fft_plan PUBLIC fft_core)
