/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fftoperation/transpose.h"
#include "base_inner_api.h"
#include "utils/ops_base.h"

using namespace AsdSip;

constexpr int DIM_TWO = 2;
Transpose::Transpose(int axis0, int axis1, Mki::SVector<int64_t> dims) : axis0_{axis0}, axis1_{axis1}, dims_{dims} {}

bool Transpose::init() { return true; }

// only for complex64
void Transpose::Run(Tensor &input, Tensor &output, void *stream, workspace::Workspace &workspace)
{
    input.desc.dtype = Mki::TensorDType::TENSOR_DTYPE_FLOAT;
    input.desc.dims.push_back(DIM_TWO);

    int axis0 = axis0_;
    if (axis0_ < 0) {
        axis0 = static_cast<int>(input.desc.dims.size()) - 1 + axis0_;
    }
    int axis1 = axis1_;
    if (axis1_ < 0) {
        axis1 = static_cast<int>(input.desc.dims.size()) - 1 + axis1_;
    }

    uint8_t *deviceBuffer = (uint8_t *)workspace.allocate(ASYNC_WORKSPACE_SIZE);

    Mki::SVector<int64_t> shapes = input.desc.dims;
    Mki::SVector<int64_t> strides(shapes.size(), 1);
    for (int64_t i = static_cast<int64_t>(shapes.size()) - 2; i >= 0; i--) {
        strides[i] = shapes[i + 1] * strides[i + 1];
    }
    int64_t shapeTmp = shapes[axis0];
    int64_t stridesTmp = strides[axis0];
    shapes[axis0] = shapes[axis1];
    strides[axis0] = strides[axis1];
    shapes[axis1] = shapeTmp;
    strides[axis1] = stridesTmp;
    AsStrided(input, output, shapes, strides, stream, 0, deviceBuffer);
    workspace.recycleLast();

    output.desc.dtype = Mki::TensorDType::TENSOR_DTYPE_COMPLEX64;
    output.desc.dims.erase(output.desc.dims.end() - 1);
}

size_t Transpose::EstimateWorkspaceSize()
{
    return getAlignedSize(ASYNC_WORKSPACE_SIZE);
}

void Transpose::Run(void *input, void *output, void *stream, workspace::Workspace &workspace)
{
    Mki::SVector<int64_t> tmpDims = dims_;
    tmpDims.push_back(DIM_TWO);

    int axis0 = axis0_;
    if (axis0_ < 0) {
        axis0 = static_cast<int>(tmpDims.size()) - 1 + axis0_;
    }
    int axis1 = axis1_;
    if (axis1_ < 0) {
        axis1 = static_cast<int>(tmpDims.size()) - 1 + axis1_;
    }

    uint8_t *deviceBuffer = (uint8_t *)workspace.allocate(ASYNC_WORKSPACE_SIZE);

    Tensor tensorIn;
    Tensor tensorOut;
    tensorIn.desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND, tmpDims, {}, 0};
    tensorIn.data = input;
    tensorOut.data = output;

    Mki::SVector<int64_t> strides(tmpDims.size(), 1);
    for (int64_t i = static_cast<int64_t>(tmpDims.size()) - 2; i >= 0; i--) {
        strides[i] = tmpDims[i + 1] * strides[i + 1];
    }
    int64_t shapeTmp = tmpDims[axis0];
    int64_t stridesTmp = strides[axis0];
    tmpDims[axis0] = tmpDims[axis1];
    strides[axis0] = strides[axis1];
    tmpDims[axis1] = shapeTmp;
    strides[axis1] = stridesTmp;
    AsStrided(tensorIn, tensorOut, tmpDims, strides, stream, 0, deviceBuffer);

    workspace.recycleLast();
}