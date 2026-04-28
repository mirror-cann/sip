/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_UNIT_TEST_COMMON_H
#define ASCEND_UNIT_TEST_COMMON_H

#include <string>
#include "mki/utils/SVector/SVector.h"

int64_t Prod(const Mki::SVector<int64_t> &vec);

/**
 * @brief 对字符串进行 shell 单引号包裹，防止路径中的空格或特殊字符导致 system() 调用解析失败。
 * @param s 原始字符串（如文件路径）
 * @return 被单引号包裹并转义内部单引号的字符串
 */
static inline std::string ShellQuote(const std::string& s)
{
    std::string result = "'";
    for (char c : s) {
        if (c == '\'') {
            result += "'\\''";
        } else {
            result += c;
        }
    }
    result += "'";
    return result;
}
#endif  // ASCEND_UNIT_TEST_COMMON_H
