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
#include "test_util/util.cpp"
#include "base_api.h"
#include "blas_api.h"
#include "utils/mem_base_inner.h"

using namespace AsdSip;
using namespace Mki;

namespace {
int AclInit(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，acl初始化
    // auto ret = aclInit(nullptr);
    // CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    auto ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

} // namespace

TEST(TestBLasAip, TestSetStreamCaseNull)
{   
    int deviceId = 0;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    asdBlasHandle handle;
    asdBlasCreate(handle);
    asdBlasMakeCopyPlan(handle);
    auto result = asdBlasSetStream(handle, nullptr);

    asdBlasDestroy(handle);
    auto ret = aclrtDestroyStream(stream);
    ASDSIP_LOG_IF(ret != 0, ERROR) << "aclrtDestroyStream fail";
    ret = aclrtResetDevice(deviceId);
    ASDSIP_LOG_IF(ret != 0, ERROR) << "aclrtResetDevice fail";
    ASSERT_EQ(result == AsdSip::ErrorType::ACL_SUCCESS, false);
}

TEST(TestBLasAip, TestSetStreamCaseInvalidHandle)
{   
    int deviceId = 0;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    asdBlasHandle handle;
    asdBlasCreate(handle);
    asdBlasMakeCopyPlan(handle);
    auto result = asdBlasSetStream(&deviceId, stream);

    asdBlasDestroy(handle);
    auto ret = aclrtDestroyStream(stream);
    ASDSIP_LOG_IF(ret != 0, ERROR) << "aclrtDestroyStream fail";
    ret = aclrtResetDevice(deviceId);
    ASDSIP_LOG_IF(ret != 0, ERROR) << "aclrtResetDevice fail";
    ASSERT_EQ(result == AsdSip::ErrorType::ACL_SUCCESS, false);
}

TEST(TestBLasAip, TestAsdBlasDestroyInvalidHandle)
{   
    int deviceId = 0;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    asdBlasHandle handle;
    asdBlasCreate(handle);
    asdBlasMakeCopyPlan(handle);
    asdBlasSetStream(handle, stream);

    auto result = asdBlasDestroy(&deviceId);
    asdBlasDestroy(handle);
    auto ret = aclrtDestroyStream(stream);
    ASDSIP_LOG_IF(ret != 0, ERROR) << "aclrtDestroyStream fail";
    ret = aclrtResetDevice(deviceId);
    ASDSIP_LOG_IF(ret != 0, ERROR) << "aclrtResetDevice fail";
    ASSERT_EQ(result == AsdSip::ErrorType::ACL_SUCCESS, false);
}