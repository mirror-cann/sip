/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TYPE_DEF
#define TYPE_DEF

#ifdef ASCEND910B1
#define CORENUM 20
#elif ASCEND910B2
#define CORENUM 30
#elif ASCEND910B3
#define CORENUM 20
#else
#define CORENUM 20
#endif

typedef enum { ASCBLAS_SIDE_LEFT, ASCBLAS_SIDE_RIGHT } ascblasSideMode_t;

typedef enum { ASCBLAS_OP_N, ASCBLAS_OP_T, ASCBLAS_OP_C } ascblasOperation_t;

typedef enum { ASCBLAS_FILL_MODE_LOWER, ASCBLAS_FILL_MODE_UPPER, ASCBLAS_FILL_MODE_FULL } ascblasFillMode_t;

typedef enum { ASCBLAS_DIAG_NON_UNIT, ASCBLAS_DIAG_UNIT } ascblasDiagType_t;

typedef struct {
    float real;
    float imag;
} ascComplex;

#endif
