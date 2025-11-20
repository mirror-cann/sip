/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <vector>
#include <memory>
#include <functional>

#include <mki/tensor.h>

namespace AsdSip {

using Mki::Tensor;

struct FFTensor : Tensor {
    ~FFTensor();
};

struct CoeffKey {
    unsigned coreType;
    unsigned userDefine;
    std::vector<int64_t> shape;
    bool isForward;
    bool operator==(const CoeffKey &other) const;
};

struct CoeffKeyHash {
    size_t operator()(const CoeffKey &one) const;
};
}
namespace FFTensorCache {

std::shared_ptr<AsdSip::FFTensor> getCoeff(AsdSip::CoeffKey &key, std::function<AsdSip::FFTensor *()> func);

}
