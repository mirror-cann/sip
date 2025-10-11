/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "the_tensor.h"

namespace wten {

template <typename T>
class TheView {
public:
    static TheView<T> select(const TheView<T> &view, int64_t dim, int64_t index);

    static TheView<T> slice(const TheView<T> &view, int64_t dim, int64_t start, int64_t end, int64_t step);

    T &operator[](size_t index) const
    {
        size_t data_index = 0;
        for (int64_t i = static_cast<int64_t>(ndim()) - 1; i >= 0; i--) {
            if (shape_[i] == 0) {
                continue;
            }
            data_index += (index % shape_[i]) * strides_[i];
            index /= shape_[i];

            if (index == 0) {
                break;
            }
        }
        data_index += offset_;

        return data_[data_index];
    }

    TheView(const TheTensor<T> &tensor)
        : data_(tensor.data()),
          size_(tensor.size()),
          offset_(0),
          shape_(tensor.shape()),
          strides_(tensor.strides())
    {
    }

    TheView(const TheView<T> &view)
        : data_(view.data_),
          size_(view.size_),
          offset_(view.offset_),
          shape_(view.shape_),
          strides_(view.strides_)
    {
    }

    // 拷贝赋值操作符
    TheView& operator=(const TheView& other) = default;

    T *data() const
    {
        return data_;
    }

    int64_t size() const
    {
        return size_;
    }

    int64_t ndim() const
    {
        return shape_.size();
    }

    int64_t offset() const
    {
        return offset_;
    }

    const std::vector<int64_t> &shape() const
    {
        return shape_;
    }

    const std::vector<int64_t> &strides() const
    {
        return strides_;
    }

private:
    T *data_;
    int64_t size_;
    int64_t offset_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
};

}
