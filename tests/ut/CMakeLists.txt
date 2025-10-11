# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

include_directories(
    $ENV{CODE_ROOT}/3rdparty/mki/include/mki/utils/platform
    $ENV{MIK_CODE}/3rdparty/nlohmannJson/include
    $ENV{CODE_ROOT}/core/include
    $ENV{CODE_ROOT}/core/utils/include
    ${PROJECT_SOURCE_DIR}/ops
)

add_subdirectory(framework)
add_subdirectory(unittest)