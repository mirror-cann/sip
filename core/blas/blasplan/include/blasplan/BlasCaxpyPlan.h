/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#pragma once

#include "BlasPlan.h"


namespace AsdSip {
class BlasCaxpyPlan : public BlasPlan {
public:
    Mki::Tensor augTensor;
    aclTensor *augAclTensor = nullptr;

public:
    BlasCaxpyPlan();
    AsdSip::AspbStatus CreateTensor() override;
    AsdSip::AspbStatus FreeTensor() override;
    ~BlasCaxpyPlan() override;
};
}
