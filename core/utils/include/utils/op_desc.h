/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_OPDESC_H
#define ASDSIP_OPDESC_H
#include <string>
#include <mki/utils/any/any.h>

namespace AsdSip {
struct OpDesc {
    int opId = 0;
    std::string opName;
    Mki::Any specificParam;
    std::string ToString() const;
    std::string ParamToJsonString() const;
};
}  // namespace AsdSip

#endif
