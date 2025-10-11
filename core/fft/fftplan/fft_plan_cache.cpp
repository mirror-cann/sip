/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fftplan/fft_plan_cache.h"

namespace FFTPlanCache {

static std::unordered_map<AsdSip::asdFftHandle, std::unique_ptr<AsdSip::FFTPlan>> plans;

AsdSip::asdFftHandle makePlan()
{
    AsdSip::asdFftHandle handle = new AsdSip::FFTPlan{};

    plans.insert({handle, std::unique_ptr<AsdSip::FFTPlan>(static_cast<AsdSip::FFTPlan *>(handle))});
    return handle;
}

bool doesPlanExist(AsdSip::asdFftHandle &handle)
{
    if (plans.find(handle) == plans.end()) {
        return false;
    }

    return true;
}

AsdSip::FFTPlan &getPlan(AsdSip::asdFftHandle &handle)
{
    return *(plans.at(handle));
}

void destroyPlan(AsdSip::asdFftHandle &handle)
{
    plans.erase(handle);
}

}
