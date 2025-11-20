/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENV_H
#define ENV_H

#include <cstdlib>
#include <cstring>
#include "utils/assert.h"
#include "log/log.h"

namespace AsdSip {
constexpr size_t MAX_ENV_STRING_LEN = 12800;

inline const char *GetEnv(const char *name)
{
    ASDSIP_CHECK(name != nullptr, "env name is nullptr!", return nullptr);
    const char *env = std::getenv(name);
    if (env != nullptr && strlen(env) <= MAX_ENV_STRING_LEN) {
        return env;
    }
    ASDSIP_LOG(WARN) << "env " << name << " is too long or not exist!";
    return nullptr;
}
} // namespace AsdSip

#endif // ENV_H