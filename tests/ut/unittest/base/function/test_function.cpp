/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include "utils/mem_base_inner.h"
#include "log/log.h"
#include "base_api.h"
#include "utils/ops_base.h"
#include "test_util/util.cpp"
#include "conj.h"

using namespace AsdSip;
using namespace Mki;

TEST(TestFunction, TestRunAsdOpsWithStreameNull)
{
    int64_t m = 4;
    int64_t n = 6;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m, n}});

    TensorContext context;

    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);
    OpTestMallocInTensors(inTensorDescs, context);

    Tensor input = context.inTensors[0];
    Tensor output = context.inTensors[1];

    OpDesc opDesc;
    opDesc.opName = "ConjOperation";
    AsdSip::OpParam::Conj param;
    opDesc.specificParam = param;

    SVector<Tensor> inTensors = {input};
    SVector<Tensor> outTensors = {output};

    Status status = RunAsdOps(nullptr, opDesc, inTensors, outTensors);
    OpTestEnd(deviceId, context, stream);
    ASSERT_EQ(status.Ok(), false);
}

TEST(TestFunction, TestRunAsdOpsV2WithStreamNull)
{
    Status status;
    int64_t m = 4;
    int64_t n = 6;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m, n}});

    aclTensorContext context;

    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);
    OpTestMallocInAclTensors(inTensorDescs, context);

    aclTensor * input = context.aclInTensors[0];
    aclTensor * output = context.aclInTensors[1];

    OpDesc opDesc;
    opDesc.opName = "ConjOperation";
    AsdSip::OpParam::Conj param;
    opDesc.specificParam = param;

    SVector<aclTensor *> inTensors = {input};
    SVector<aclTensor *> outTensors = {output};

    status = RunAsdOpsV2(nullptr, opDesc, inTensors, outTensors);
    OpTestEnd(deviceId, context, stream);
    ASSERT_EQ(status.Ok(), false);
}

TEST(TestFunction, TestToAclTensorWithStrideSizeNotEqualShapeSize)
{
    Status status;
    int64_t m = 4;
    int64_t n = 6;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, n}});

    TensorContext context;
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);
    OpTestMallocInTensors(inTensorDescs, context);

    Tensor input = context.inTensors[0];
    aclTensor *inAclTensor = nullptr;
    std::vector<int64_t> stride = {1};
    status = toAclTensor(input, inAclTensor, stride);

    OpTestEnd(deviceId, context, stream);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestFunction, TestMallocOutTensorWihtInvalidOutTensors01)
{
    Status status;
    int64_t m = 4;
    int64_t n = 6;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m, n}});

    TensorContext context;

    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);
    OpTestMallocInTensors(inTensorDescs, context);

    Tensor input = context.inTensors[0];

    OpDesc opDesc;
    opDesc.opName = "ConjOperation";
    AsdSip::OpParam::Conj param;
    opDesc.specificParam = param;

    SVector<Tensor> inTensors = {input};
    SVector<Tensor> outTensors = {};

    status = RunAsdOps(stream, opDesc, inTensors, outTensors);
    OpTestEnd(deviceId, context, stream);
    ASSERT_EQ(status.Ok(), true);
}


TEST(TestFunction, TestMallocOutTensorWihtInvalidOutTensors02)
{
    Status status;
    int64_t m = 4;
    int64_t n = 6;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m, n}});

    TensorContext context;

    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);
    OpTestMallocInTensors(inTensorDescs, context);

    Tensor input = context.inTensors[0];
    Tensor output1 = context.inTensors[1];
    Tensor output2 = context.inTensors[2];

    OpDesc opDesc;
    opDesc.opName = "ConjOperation";
    AsdSip::OpParam::Conj param;
    opDesc.specificParam = param;

    SVector<Tensor> inTensors = {input};
    SVector<Tensor> outTensors = {output1, output2};

    status = RunAsdOps(stream, opDesc, inTensors, outTensors);
    OpTestEnd(deviceId, context, stream);
    ASSERT_EQ(status.Ok(), false);
}

TEST(TestFunction, TestMallocOutTensorWihtInvalidInTensors01)
{
    Status status;
    int64_t m = 4;
    int64_t n = 6;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m, n}});

    TensorContext context;

    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);
    OpTestMallocInTensors(inTensorDescs, context);

    Tensor input = context.inTensors[0];
    Tensor input2 = context.inTensors[0];
    Tensor out = context.inTensors[0];
    OpDesc opDesc;
    opDesc.opName = "ConjOperation";
    AsdSip::OpParam::Conj param;
    opDesc.specificParam = param;

    SVector<Tensor> inTensors = {input, input2};
    SVector<Tensor> outTensors = {out};

    status = RunAsdOps(stream, opDesc, inTensors, outTensors);
    OpTestEnd(deviceId, context, stream);
    ASSERT_EQ(status.Ok(), false);
}