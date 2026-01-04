/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASSERT_H
#define ASSERT_H

#include "log/log.h"

namespace AsdSip {
namespace Utils {

#define ASDSIP_ECHECK(condition, logExpr, handleExpr)                                                               \
do {                                                                                                                \
    if (!(condition)) {                                                                                             \
        ASDSIP_ELOG(handleExpr) << (logExpr);                                                                       \
        return handleExpr;                                                                                          \
    }                                                                                                               \
} while (0)

#define CHECK_STATUS_WITH_ACL_RETURN(ret, funcName)                                                                 \
    if ((ret) == 0) {                                                                                               \
        ASDSIP_LOG(DEBUG) << (funcName) << " success";                                                              \
    } else {                                                                                                        \
        ASDSIP_LOG(ERROR) << (funcName) << " fail, aclError code is: " << (ret);                                    \
        return ::ACL_ERROR_INTERNAL_ERROR;                                                                            \
    }

#define ASDSIP_CHECK(condition, logExpr, handleExpr)                                                                \
do {                                                                                                                \
    if (!(condition)) {                                                                                             \
        ASDSIP_LOG(ERROR) << (logExpr);                                                                             \
        handleExpr;                                                                                                 \
    }                                                                                                               \
} while (0)

#define ASDSIP_CHECK_NO_LOG(condition, handleExpr)                                                                  \
do {                                                                                                                \
    if (!(condition)) {                                                                                             \
        handleExpr;                                                                                                 \
    }                                                                                                               \
} while (0)
}

#define ASDSIP_CHECK_WITH_NO_RETURN(condition, logExpr, handleExpr)                                                               \
do {                                                                                                                \
    if (!(condition)) {                                                                                             \
        ASDSIP_ELOG(handleExpr) << (logExpr);                                                                       \
    }                                                                                                               \
} while (0)

#define UNUSED_VALUE(x) (void)(x)
} // namespace AsdSip
#endif
