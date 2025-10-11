# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if("$ENV{TEST_TYPE}" STREQUAL "UT")
    add_library(utils STATIC the_tensor.cpp the_view.cpp ops_base.cpp common_check.cpp sip_lock.cpp)
else()
    add_library(utils STATIC the_tensor.cpp the_view.cpp ops_base.cpp common_check.cpp sip_lock.cpp log_core.cpp log_entity.cpp log_sink_file.cpp log_sink_stdout.cpp 
        log_stream.cpp
    )
endif()

if("$ENV{TEST_TYPE}" STREQUAL "UT" OR "$ENV{TEST_TYPE}" STREQUAL "FT")
    target_link_libraries(utils PUBLIC asdsip_host mki dl c_sec mmpa ascendcl nnopbase)
else()
    target_link_libraries(utils PUBLIC asdsip_host 
        -Wl,--whole-archive
        mki_static 
        -Wl,--no-whole-archive
        dl c_sec mmpa ascendcl nnopbase runtime
        -Wl,--exclude-libs=ALL
    )
endif()

target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(utils PUBLIC $ENV{ASCEND_HOME_PATH}/include)
target_include_directories(utils PUBLIC $ENV{ASCEND_HOME_PATH}/include/aclnn)

install(FILES include/utils/mem_base.h include/utils/aspb_status.h DESTINATION include/utils)
