# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# mki: an existing shared target
file(GLOB_RECURSE MKI_LIBS "${MKI_PACKAGE_DIR}/lib**/libmki.so")
list(GET MKI_LIBS 0 MKI_LIB)

add_library(mki SHARED IMPORTED GLOBAL)
set_property(TARGET mki PROPERTY IMPORTED_LOCATION ${MKI_LIB})

target_include_directories(mki INTERFACE ${MKI_PACKAGE_DIR}/include)

# mki_static: an existing shared target
file(GLOB_RECURSE MKI_STATIC_LIBS "${MKI_PACKAGE_DIR}/lib**/libmki_static.a")
list(GET MKI_STATIC_LIBS 0 MKI_STATIC_LIB)

add_library(mki_static STATIC IMPORTED GLOBAL)
set_property(TARGET mki_static PROPERTY IMPORTED_LOCATION ${MKI_STATIC_LIB})

target_include_directories(mki_static INTERFACE ${MKI_PACKAGE_DIR}/include)
