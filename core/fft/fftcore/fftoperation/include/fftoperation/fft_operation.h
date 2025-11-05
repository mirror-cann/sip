/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FFTOPERATION__
#define __FFTOPERATION__

#include <mki/tensor.h>
#include "aclnn/acl_meta.h"
#include "utils/workspace.h"
#include "mki/kernel.h"

using Tensor = Mki::Tensor;

class FftOperation {
public:
    virtual ~FftOperation() = default;
    virtual bool init() = 0;
    virtual void Run(Tensor &input, Tensor &output, void *stream, workspace::Workspace &workspace) = 0;

    virtual void Run(void *input, void *output, void *stream, workspace::Workspace &workspace)
    {
        const Mki::KernelInfo &kernelInfo = kernel->GetKernelInfo();
        size_t bufferSize = kernelInfo.GetTotalScratchSize();
        runInfo.SetScratchDeviceAddr((uint8_t *)workspace.allocate(bufferSize));
        runInfo.SetStream(stream);
        launchParam.GetInTensor(0).data = input;
        launchParam.GetOutTensor(0).data = output;
        kernel->Run(launchParam, runInfo);
        workspace.recycleLast();
    }

    virtual size_t EstimateWorkspaceSize()
    {
        return 0;
    }
    virtual void allocateWorkspace() {}
    virtual void recycleWorkspace() {}

protected:
    std::unique_ptr<Mki::Kernel> kernel;
    Mki::RunInfo runInfo;
    Mki::LaunchParam launchParam;
};

#endif
