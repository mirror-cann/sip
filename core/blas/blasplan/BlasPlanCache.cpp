/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <memory>
#include <unordered_map>
#include "include/blasplan/BlasPlanCache.h"

namespace BlasPlanCache {

static std::unordered_map<AsdSip::asdBlasHandle, std::unique_ptr<AsdSip::BlasPlan>> plans;

void MakePlan(AsdSip::asdBlasHandle &handle, AsdSip::BlasPlan *plan)
{
    plans.insert({handle, std::unique_ptr<AsdSip::BlasPlan>(plan)});
}

AsdSip::asdBlasHandle InitHandle()
{
    AsdSip::asdBlasHandle handle = new int();
    return handle;
}

bool doesPlanExist(AsdSip::asdBlasHandle &handle)
{
    if (plans.find(handle) == plans.end()) {
        return false;
    }

    return true;
}

AsdSip::BlasPlan &getPlan(AsdSip::asdBlasHandle &handle)
{
    if (plans.at(handle) == nullptr) {
        throw std::runtime_error("BlasPlan is nullptr.");
    } else {
        return *(plans.at(handle));
    }
}

void destroy_plan(AsdSip::asdBlasHandle &handle)
{
    plans.erase(handle);
}

}
