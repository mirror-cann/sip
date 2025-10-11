# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_subdirectory(convolve)

set(operations_objects ${operations_objects} ${ops_objects} PARENT_SCOPE)

set_source_files_properties(${BINARY_SRC_LIST} PROPERTIES GENERATED TRUE)

add_custom_command(
    OUTPUT ${BINARY_SRC_LIST}
    DEPENDS ${BINARY_TARGET_LIST}
    COMMAND python3 ${MKI_PACKAGE_DIR}/scripts/build_util.py --binary_dir ${CMAKE_BINARY_DIR} --op_type "ascendc"
)

add_custom_target(FILTER_BINARY_SRC_TARGET ALL
    DEPENDS ${BINARY_SRC_LIST}
)

add_library(filter_obj OBJECT ${BINARY_SRC_LIST})
add_dependencies(filter_obj FILTER_BINARY_SRC_TARGET)
