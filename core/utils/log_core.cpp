/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdlib>
#include <string>
#include <cstring>
#include <unordered_map>
#include <iostream>
#include <algorithm>
#include "utils/env.h"
#include "log/log_sink_stdout_sip.h"
#include "log/log_sink_file_sip.h"
#include "log/log.h"
#include "log/log_core_sip.h"

namespace AsdSip {
static bool GetLogToStdoutFromEnv()
{
    const char *envLogToStdout = std::getenv("ASCEND_SLOG_PRINT_TO_STDOUT");
    return envLogToStdout != nullptr && strlen(envLogToStdout) <= MAX_ENV_STRING_LEN &&
            strcmp(envLogToStdout, "1") == 0;
}

static bool GetLogToFileFromEnv()
{
    const char *envLogToFile = std::getenv("ASCEND_PROCESS_LOG_PATH");
    return envLogToFile != nullptr && strlen(envLogToFile) <= MAX_ENV_STRING_LEN;
}

static LogLevel GetLogLevelFromEnv()
{
    /* 设置应用类日志的日志级别及各模块日志级别，仅支持调试日志。
    取值为：
        0：对应DEBUG级别。
        1：对应INFO级别。
        2：对应WARNING级别。
        3：对应ERROR级别，默认为ERROR级别。
        4：对应NULL级别，不输出日志。
        其他值为非法值。 */
    // LogLevel {DEBUG, INFO, WARN, ERROR } FATAL对应NULL级别，不输出日志。
    const char *env = std::getenv("ASCEND_GLOBAL_LOG_LEVEL");
    if (env == nullptr || strlen(env) > MAX_ENV_STRING_LEN) {
        return LogLevel::WARN;
    }
    std::string envLogLevel(env);
    std::transform(envLogLevel.begin(), envLogLevel.end(), envLogLevel.begin(), ::toupper);
    static std::unordered_map<std::string, LogLevel> levelMap{{"0", LogLevel::DEBUG},
        {"1", LogLevel::INFO},
        {"2", LogLevel::WARN},
        {"3", LogLevel::ERROR},
        {"4", LogLevel::FATAL}};
    auto levelIt = levelMap.find(envLogLevel);
    // LogLevel::FATAL对应NULL级别，不输出日志。
    return levelIt != levelMap.end() ? levelIt->second : LogLevel::ERROR;
}

LogCoreSip::LogCoreSip()
{
    level_ = GetLogLevelFromEnv();
    if (strcmp(LogLevelToString(level_), "FATAL") == 0) {
        return;
    }
    if (GetLogToStdoutFromEnv()) { // print but not to file
        AddSink(std::make_shared<LogSinkStdoutSip>());
    } else if (!GetLogToStdoutFromEnv() && GetLogToFileFromEnv()) { // not print but to file
        AddSink(std::make_shared<AsdSip::LogSinkFileSip>());
    }
}

LogCoreSip &LogCoreSip::InstanceSip()
{
    static LogCoreSip logCore;
    return logCore;
}

LogLevel LogCoreSip::GetLogLevel() const { return level_; }

void LogCoreSip::SetLogLevel(LogLevel level) { level_ = level; }

void LogCoreSip::LogSip(const char *log, uint64_t logLen)
{
    for (auto &sink : sinks_) {
        sink->LogSip(log, logLen);
    }
}

void LogCoreSip::DeleteLogFileSink()
{
    if (sinks_.size() == 0) {
        ASDSIP_LOG(ERROR) << "sink_ is empty";
        return;
    }
    sinks_.pop_back();
}

void LogCoreSip::AddSink(std::shared_ptr<LogSinkSip> sink)
{
    if (sink == nullptr) {
        ASDSIP_LOG(ERROR) << "sink is NULL";
        return;
    }
    sinks_.push_back(sink);
}

const std::vector<std::shared_ptr<LogSinkSip>> &LogCoreSip::GetAllSinks() const { return sinks_; }
} // namespace AsdSip
