/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
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

#include "fftcore/fft_r2c_core.h"

using namespace AsdSip;

bool FftR2CCore::PreAllocateInDevice()
{
    return FftCoreMixBase::PreAllocateX2XInDevice();
}

std::string FftR2CCore::GetOpName()
{
    return std::string("FftR2COperation");
}

void FftR2CCore::InitLaunchParam()
{
    OpParam::FftAllMix param = {parity,
                                problemDesc.nDoing,
                                aivSplitWay,
                                problemDesc.batch,
                                nFftC2c,
                                radixVec.size(),
                                1 - int(problemDesc.forward),
                                workspaceInputSize,
                                workspaceOutputSize,
                                workspaceSyncSize,
                                workspaceC2cOutputSize,
                                workspaceAuxilSize};

    Tensor tensorIn;
    Tensor tensorOut;
    // need check if input shape is right
    int64_t inputN = problemDesc.nDoing;
    tensorIn.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {problemDesc.batch, inputN}, {}, 0};
    tensorIn.dataSize = static_cast<size_t>(problemDesc.batch) * static_cast<size_t>(inputN)
                        * GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_FLOAT);

    int64_t outputN = problemDesc.nDoing / 2 + 1;
    tensorOut.dataSize = static_cast<size_t>(problemDesc.batch) * static_cast<size_t>(outputN)
                         * GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64);

    launchParam.SetParam(param);
    launchParam.AddInTensor(tensorIn);
    launchParam.AddInTensor(*inputIndex);
    launchParam.AddInTensor(*aTable);
    launchParam.AddInTensor(*bTable);
    launchParam.AddInTensor(*outputIndex);
    launchParam.AddInTensor(*dftMatrixArray);
    launchParam.AddInTensor(*twMatrixArray);
    launchParam.AddInTensor(*radixList);
    launchParam.AddOutTensor(tensorOut);
}