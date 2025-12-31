/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <numeric>
#include <tuple>
#include "utils/operations.h"
#include "utils/rotate.h"

namespace rotate {

constexpr int K_COMPLEX_SIZE = 2;

constexpr int K_FACTOR_2 = 2;

constexpr int K_DIM_2 = 2;

constexpr int K_ROTATION_DIMS = 4;

using FftMode = AsdSip::FftMode;

std::tuple<int64_t, int64_t, int64_t, int64_t> computeOutShape(const std::vector<int64_t> &factors,
                                                               int64_t index, FftMode fftMode)
{
    int64_t factor = factors[index];
    int64_t outN = ((fftMode == FftMode::r2cOneside) && (index == (static_cast<int64_t>(factors.size()) - 1))) ?
                               (factor / K_FACTOR_2) + 1 :
                               factor;
    int64_t outComplex =
        ((fftMode == FftMode::c2r) && (index == (static_cast<int64_t>(factors.size()) - 1))) ?
        1 : K_COMPLEX_SIZE;
    int64_t inN = factor;
    int64_t inComplex =
        ((fftMode == FftMode::r2cOneside || fftMode == FftMode::r2cBothside) && (index == 0)) ?
        1 : K_COMPLEX_SIZE;

    return std::make_tuple(outN, outComplex, inN, inComplex);
}

template <typename T>
void CopyQuarter(wten::TheTensor<T> &dst, wten::TheTensor<T> &src, int64_t prevN,
                 int64_t factor, bool fullIn, bool fullOut, int64_t x, int64_t y)
{
    if (fullIn && fullOut) {
        auto dstView = wten::TheView<T>::slice(dst, 0, 0, prevN, 1);
        dstView = wten::TheView<T>::slice(dstView, 1, 0, factor, 1);
        dstView = wten::TheView<T>::select(dstView, K_DIM_2, x);
        dstView = wten::TheView<T>::select(dstView, K_DIM_2, y);
        dstView = wten::TheView<T>::slice(dstView, K_DIM_2, 0, factor, 1);

        wten::copy(src, dstView);
    } else if (!fullIn) {
        auto dstView = wten::TheView<T>::slice(dst, 0, 0, prevN, 1);
        dstView = wten::TheView<T>::slice(dstView, 1, 0, factor, 1);
        dstView = wten::TheView<T>::select(dstView, K_DIM_2, x);
        dstView = wten::TheView<T>::select(dstView, K_DIM_2, y);
        dstView = wten::TheView<T>::slice(dstView, K_DIM_2, 0, (factor / K_FACTOR_2) + 1, 1);

        auto srcView = wten::TheView<T>::slice(src, K_DIM_2, 0, (factor / K_FACTOR_2) + 1, 1);
        wten::copy(srcView, dstView);
    } else if (!fullOut) {
        auto dstView = wten::TheView<T>::slice(dst, 0, 0, prevN, 1);
        dstView = wten::TheView<T>::slice(dstView, 1, 0, (factor / K_FACTOR_2) + 1, 1);
        dstView = wten::TheView<T>::select(dstView, K_DIM_2, x);
        dstView = wten::TheView<T>::select(dstView, K_DIM_2, y);
        dstView = wten::TheView<T>::slice(dstView, K_DIM_2, 0, factor, 1);

        auto srcView = wten::TheView<T>::slice(src, 1, 0, (factor / K_FACTOR_2) + 1, 1);
        wten::copy(srcView, dstView);
    }
}

void copyAll(wten::TheTensor<float> &rotateMatrix, wten::TheTensor<double> &theta, int64_t prevN,
             int64_t factor, bool isForward, bool isInComplex, bool isOutComplex, bool isFullIn, bool isFullOut)
{
    wten::TheTensor<double> triangleD{theta.shape()};
    wten::TheTensor<float> triangleF{theta.shape()};

    wten::cos(theta, triangleD);
    wten::cast(triangleD, triangleF);
    CopyQuarter(rotateMatrix, triangleF, prevN, factor, isFullIn, isFullOut, 0, 0);
    if (isInComplex && isOutComplex) {
        CopyQuarter(rotateMatrix, triangleF, prevN, factor, isFullIn, isFullOut, 1, 1);
    }
    wten::sin(theta, triangleD);
    wten::cast(triangleD, triangleF);
    if (isForward) {
        if (isOutComplex) {
            CopyQuarter(rotateMatrix, triangleF, prevN, factor, isFullIn, isFullOut, 1, 0);
        }
        if (isInComplex) {
            wten::neg(triangleF, triangleF);
            CopyQuarter(rotateMatrix, triangleF, prevN, factor, isFullIn, isFullOut, 0, 1);
        }
    } else {
        if (isInComplex) {
            CopyQuarter(rotateMatrix, triangleF, prevN, factor, isFullIn, isFullOut, 0, 1);
        }
        if (isOutComplex) {
            wten::neg(triangleF, triangleF);
            CopyQuarter(rotateMatrix, triangleF, prevN, factor, isFullIn, isFullOut, 1, 0);
        }
    }
}

wten::TheTensor<float> OneRotateMatrix(std::vector<int64_t> factors, int64_t index, FftMode fftMode, bool isForward)
{
    int64_t prevN = 1;
    auto itEnd = factors.begin() + index;
    for (auto it = factors.begin(); it < itEnd; it++) {
        prevN *= *it;
    }

    int64_t factor = factors[index];

    // compute theta
    auto firstDim = wten::arange<int64_t>(0, prevN, 1);
    auto secondDim = wten::arange<int64_t>(0, prevN * factor, prevN);
    auto thirdDim = wten::arange<int64_t>(0, -factor, -1);

    firstDim.reshape({prevN, 1});
    secondDim.reshape({1, factor});
    thirdDim.reshape({1, factor});

    auto thetaI = wten::outerAdd(firstDim, secondDim);
    thetaI.reshape({(prevN * factor), 1});
    thetaI = wten::outerMul(thetaI, thirdDim);

    firstDim.free_data();
    secondDim.free_data();
    thirdDim.free_data();

    thetaI.reshape({prevN, factor, factor});
    wten::modScalar(thetaI, prevN * factor, thetaI);
    auto theta = wten::cast<int64_t, double>(thetaI);
    wten::mulScalar(theta, M_PI * K_FACTOR_2 / (prevN * factor), theta);

    thetaI.free_data();

    auto [outN, outComplex, inN, inComplex] = computeOutShape(factors, index, fftMode);

    std::vector<int64_t> rotateShape{prevN, outN, outComplex, inComplex, inN};
    wten::TheTensor<float> rotateMatrix{rotateShape};

    // compute rotate
    copyAll(rotateMatrix, theta, prevN, factor, isForward,
            inComplex == K_COMPLEX_SIZE, outComplex == K_COMPLEX_SIZE, inN == factor, outN == factor);

    // transpose dims
    std::vector<int64_t> splitShape(index + K_ROTATION_DIMS);
    std::copy(factors.begin(), factors.begin() + index, splitShape.rbegin() + K_ROTATION_DIMS);
    std::copy(rotateShape.begin() + 1, rotateShape.end(), splitShape.begin() + index);

    std::vector<int64_t> permuteShape(index + K_ROTATION_DIMS);
    std::iota(permuteShape.rbegin() + K_ROTATION_DIMS, permuteShape.rend(), int64_t{0});
    std::iota(permuteShape.begin() + index, permuteShape.end(), int64_t{index});

    rotateMatrix.reshape(splitShape);
    rotateMatrix = wten::transpose(rotateMatrix, permuteShape);

    std::vector<int64_t> reshapeShape{prevN, outN * outComplex, inComplex * inN};
    rotateMatrix.reshape(reshapeShape);
    return rotateMatrix;
}

}
