/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "log/log_entity.h"
#include <vector>
#include <string>

namespace AsdSip {
const char* LogLevelToString(LogLevel level)
{
    static std::vector<std::string> levelStrs = {"FATAL", "DEBUG", "INFO", "WARN", "ERROR"};
    size_t levelInt = static_cast<size_t>(level);
    return (levelInt < levelStrs.size() ? levelStrs[levelInt] : "UNKNOWN").c_str();
}

const int LogErrCodeToInt(ErrorType errCode)
{
    return static_cast<int>(errCode);
}
} // namespace AsdSip
