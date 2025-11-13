/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASPB_STATUS_H
#define ASPB_STATUS_H

#include <cstdint>

namespace AsdSip {
using AspbStatus = int32_t;

enum ErrorType : int {
    ACL_SUCCESS = 0,
    ACL_ERROR_INVALID_PARAM = 100000,
    ACL_ERROR_OP_INPUT_NOT_MATCH = 100021,
    ACL_ERROR_OP_OUTPUT_NOT_MATCH = 100022,
    ACL_ERROR_UNSUPPORTED_DATA_TYPE = 100026,
    ACL_ERROR_FORMAT_NOT_MATCH = 100027,
    ACL_ERROR_API_NOT_SUPPORT = 200001,
    ACL_ERROR_INTERNAL_ERROR = 500000
};
}  // namespace AsdSip

#endif
