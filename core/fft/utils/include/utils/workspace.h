/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FFT_WORKSPACE_H__
#define __FFT_WORKSPACE_H__

#include <vector>
#include <cstddef>
#include <mki/tensor.h>
#include "log/log.h"

using Mki::Tensor;

constexpr int K_BLOCK_SIZE = 32;

namespace workspace {

class Workspace {
public:
    Workspace() = default;
    Workspace(const Tensor &dataSegment);
    Workspace(void *workSpace);
    bool isInitialized() const;
    void Reset();
    void *allocate(size_t dataSize);
    void recycleLast();

private:
    void *dataPtr_ = nullptr;
    size_t dataSize_ = 0;
    std::vector<size_t> offsets_ = {0};
};

}

inline size_t getAlignedSize(size_t size)
{
    if (size > (SIZE_MAX - K_BLOCK_SIZE + 1)) {
        ASDSIP_LOG(ERROR) << "The size in getAlignedSize is overflowing.";
        throw std::runtime_error("The size in getAlignedSize is overflowing.");
    }
    return (size + K_BLOCK_SIZE - 1) / K_BLOCK_SIZE * K_BLOCK_SIZE;
}

#endif
