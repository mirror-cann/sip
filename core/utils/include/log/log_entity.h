/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LOG_ENTITY_H
#define LOG_ENTITY_H
#include <chrono>
#include <string>
#include "utils/aspb_status.h"

namespace AsdSip {
enum class LogLevel { FATAL = 0, DEBUG, INFO, WARN, ERROR };

const char *LogLevelToString(LogLevel level);
const int LogErrCodeToInt(ErrorType errCode);

struct LogEntity {
    std::chrono::system_clock::time_point time = std::chrono::system_clock::now();
    long threadId = 0;
    LogLevel level = LogLevel::FATAL;
    ErrorType errCode = ErrorType::ACL_SUCCESS;
    const char *fileName = nullptr;
    int line = 0;
    const char *funcName = nullptr;
};
} // namespace AsdSip
#endif
