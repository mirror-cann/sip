/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LOG_SINK_H
#define LOG_SINK_H
#include "log/log_entity.h"

namespace AsdSip {
class LogSinkSip {
public:
    LogSinkSip() = default;
    virtual ~LogSinkSip() = default;
    virtual void LogSip(const char *log, uint64_t logLen) = 0;
};
} // namespace AsdSip
#endif
