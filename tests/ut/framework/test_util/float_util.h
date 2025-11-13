/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FLOATUTIL_H
#define FLOATUTIL_H

#include <vector>
#include "log/log.h"
#include "mki/utils/status/status.h"

class FloatUtil {
public:
    /**
     * @brief 指定精度下判断两float数据是否相同
     * @param excepted      期望数据
     * @param actual        真实计算数据
     * @param atol          绝对误差的阈值，例如0.001
     * @param rtol          相对误差的阈值，例如0.001
     * @return 数据相同返回true，不同返回false
    */
    static bool FloatJudgeEqual(float expected, float actual, float atol, float rtol);

    /**
     * @brief 生成指定大小范围的float类型随机数
     * @param lower         随机数大小下限
     * @param upper         随机数大小上限
     * @param code          输出，生成的随机数vector，数据类型转为T
    */
    template <class T>
    static void GenerateCode(float lower, float upper, std::vector<T> &code);
};

#endif
