/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_UNIT_TEST_DEVICE_VERSION_CHECK_H
#define ASCEND_UNIT_TEST_DEVICE_VERSION_CHECK_H

#include "log/log.h"

#define CHACK_DEVICE_VERSION_ASCEND910B()                                \
    do {                                                                 \
        if (!(IsAscend910B())) {                                         \
            ASD_LOG(WARN) << "check device, not in 910b, skip testcase"; \
            return;                                                      \
        }                                                                \
    } while (0)

#define CHACK_DEVICE_VERSION_NOT_ASCEND910B()                        \
    do {                                                             \
        if ((IsAscend910B())) {                                      \
            ASD_LOG(WARN) << "check device, in 910b, skip testcase"; \
            return;                                                  \
        }                                                            \
    } while (0)

#define CHACK_DEVICE_VERSION_ASCEND910A()                                \
    do {                                                                 \
        if (!(IsAscend910A())) {                                         \
            ASD_LOG(WARN) << "check device, not in 910a, skip testcase"; \
            return;                                                      \
        }                                                                \
    } while (0)

#define CHACK_DEVICE_VERSION_NOT_ASCEND910A()                        \
    do {                                                             \
        if ((IsAscend910A())) {                                      \
            ASD_LOG(WARN) << "check device, in 910a, skip testcase"; \
            return;                                                  \
        }                                                            \
    } while (0)

#define CHACK_DEVICE_VERSION_ASCEND310P()                                \
    do {                                                                 \
        if (!(IsAscend310P())) {                                         \
            ASD_LOG(WARN) << "check device, not in 310p, skip testcase"; \
            return;                                                      \
        }                                                                \
    } while (0)

#define CHACK_DEVICE_VERSION_NOT_ASCEND310P()                        \
    do {                                                             \
        if ((IsAscend310P())) {                                      \
            ASD_LOG(WARN) << "check device, in 310p, skip testcase"; \
            return;                                                  \
        }                                                            \
    } while (0)

bool IsAscend310P();
bool IsAscend910A();
bool IsAscend910B();
#endif  // ASCEND_UNIT_TEST_DEVICE_VERSION_CHECK_H