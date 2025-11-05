/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mki/utils/rt/rt.h>
#include "log/log.h"
#include "ops.h"
#include "fftcore/fft_core_common_func.h"
#include "utils/ops_base.h"
#include "fft_all_mix.h"

#include "fftcore/fft_core_mix.h"

using namespace AsdSip;

bool FftCoreMix::PreAllocateInDevice()
{
    return FftCoreMixBase::PreAllocateC2CInDevice();
}

void FftCoreMix::InitParams()
{
    FftCoreMixBase::InitParams();
    workspaceC2cOutputSize = 0;
}

std::string FftCoreMix::GetOpName()
{
    return std::string("FftMixOperation");
}

void FftCoreMix::InitLaunchParam()
{
    OpParam::FftAllMix param = {-1,
                                problemDesc.nDoing,
                                aivSplitWay,
                                problemDesc.batch,
                                nFftC2c,
                                radixVec.size(),
                                1 - int(problemDesc.forward),
                                workspaceInputSize,
                                workspaceOutputSize,
                                workspaceSyncSize,
                                0,
                                workspaceAuxilSize};

    Tensor tensorIn;
    Tensor tensorOut;
    // need check if input shape is right
    unsigned inputN = problemDesc.nDoing;
    tensorIn.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {problemDesc.batch, inputN}, {}, 0};
    tensorIn.dataSize = problemDesc.batch * inputN * GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64);
    tensorOut.dataSize = problemDesc.batch * inputN * GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64);

    launchParam.SetParam(param);
    launchParam.AddInTensor(tensorIn);
    launchParam.AddInTensor(*dftMatrixArray);
    launchParam.AddInTensor(*twMatrixArray);
    launchParam.AddInTensor(*radixList);
    launchParam.AddOutTensor(tensorOut);
}
