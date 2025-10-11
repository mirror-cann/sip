# **AscendSiPBoost 信号处理加速库ST框架介绍**

## 框架目录：

├─st //st测试目录

│  ├─asdsip_api_binder.cpp //算子绑定文件

│  ├─CMakeLists.txt //编译文件



## 框架介绍：

框架结合torch.extension及pybind11的能力，将当前加速库中的tensor相关对象、元素类型等和pytorch及torch_npu对应对象及元素类型做映射，从而具备通过python访问加速库算子的能力。

### 对象映射关系介绍

**torch::Tensor <<=>> AsdOps::Tensor**

依赖类库：

```c++
#include <torch/extension.h>
#include <pybind11/pybind11.h>
```

框架提供转换函数如下：

```c++
AsdOps::Tensor torchTensor2AsdTensor(const torch::Tensor &torchTensor)
{
    AsdOps::Tensor asdTensor;
    asdTensor.data = torchTensor.data_ptr();
    asdTensor.dataSize = torchTensor.numel() * torchTensor.element_size();
    asdTensor.desc.dims.resize(torchTensor.sizes().size());

    for (uint64_t i = 0; i < torchTensor.sizes().size(); i++) {
        asdTensor.desc.dims[i] = torchTensor.sizes()[i];}

    asdTensor.desc.format = TENSOR_FORMAT_ND;

    auto it = DTYPE_MAP.find(torchTensor.scalar_type());
    if (it != DTYPE_MAP.end()) {
        asdTensor.desc.dtype = it->second;} else {
        ASD_LOG(ERROR) << "not support dtype:" << torchTensor.scalar_type();}

    return asdTensor;
}
```

此函数将输入的torchTensor（要求传入的是torch_npu的tensor）的device地址赋给asdTensor.data属性从而使得asdTensor对象直接操作device上的数据并同时把shape、dtype的属性赋给asdTensor对象。

**torch_npu::NPUStream  <<=>> AsdOps::AsdRtStream**

依赖类库：

```c++
#include <torch_npu/csrc/core/npu/NPUStream.h>
```

框架提供转换函数如下：

```c++
void *getCurrentStream()
{
    AsdOps::AsdRtDeviceGetCurrent(&global_devId);
    void *stream = c10_npu::getCurrentNPUStream(global_devId).stream();
    return stream;
}
```

此函数将当前NPU卡对应的NPUStream赋给一个void指针，AsdRtStream对象可以直接引用参与算子运算。

**torch::Dtype <<=>> AsdOps::TensorDType**

依赖类库：

```c++
#include <torch/extension.h>
#include <pybind11/pybind11.h>
```

框架提供以下映射方式如下：

```
static std::map<at::ScalarType, AsdOps::TensorDType> DTYPE_MAP = {
    {torch::ScalarType::Byte, AsdOps::TENSOR_DTYPE_UINT8},   {torch::ScalarType::Char, AsdOps::TENSOR_DTYPE_INT8},
    {torch::ScalarType::Half, AsdOps::TENSOR_DTYPE_FLOAT16}, {torch::ScalarType::Float, AsdOps::TENSOR_DTYPE_FLOAT},
    {torch::ScalarType::Int, AsdOps::TENSOR_DTYPE_INT32},    {torch::ScalarType::Long, AsdOps::TENSOR_DTYPE_INT64},
    {torch::ScalarType::BFloat16, AsdOps::TENSOR_DTYPE_BF16},
    {torch::ScalarType::ComplexFloat, AsdOps::TENSOR_DTYPE_COMPLEX64},
    {torch::ScalarType::ComplexDouble, AsdOps::TENSOR_DTYPE_COMPLEX128}
};

auto inIt = DTYPE_MAP.find(torch::python::detail::py_object_to_dtype(in_dtype));
auto in_AsdDtype = inIt -> second;
```

其中in_dtype对象使用py::object类型接收。

**complex <<=>> std::complex**

依赖类库：

```c++
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
```

框架无需做特殊处理，python侧传入complex类型，如：complex(1.0, 2.0)，此时c++侧使用std::complex类型接收即可。

**其他pybind11支持的类型映射**

可参考pybind11 include路径下的头文件：

$ENV{PYTHON_LIB_PATH}/pybind11/include/

**自定义类型映射**

pybind11框架自身提供了自定义类型映射的能力：

```c++
#include <pybind11/pybind11.h>

// 定义一个简单的 C++ 类
class MyClass {
public:
    MyClass(int value) : value_(value) {}
    int getValue() const { return value_; }

private:
    int value_;
};

namespace py = pybind11;

// 使用 PYBIND11_MODULE 宏导出 C++ 类到 Python
PYBIND11_MODULE(my_module, m) {
    py::class_<MyClass>(m, "MyClass")
        .def(py::init<int>())          // 定义构造函数
        .def("get_value", &MyClass::getValue);  // 定义成员函数
}
```

python侧使用：

```python
import my_module

obj = my_module.MyClass(42)
print(obj.get_value())  # 输出 42
```

### 自定义算子封装介绍

以matMul算子为例：

```c++
uint8_t _matMul(const torch::Tensor &inATensor, const torch::Tensor &inBTensor, 
    int m, int k, int n, torch::Tensor &outTensor)
{
    AsdOps::Tensor inAsdTensorA = torchTensor2AsdTensor(inATensor);
    AsdOps::Tensor inAsdTensorB = torchTensor2AsdTensor(inBTensor);
    AsdOps::Tensor outAsdTensor = torchTensor2AsdTensor(outTensor);
    
    AsdRtStream stream = getCurrentStream();

    Status status = matMul(inAsdTensorA, inAsdTensorB, outAsdTensor, m, k, n, stream);
    if (!status.Ok()) {
        ASD_LOG(ERROR) << "Something error.";
        return -1;
    }
    
    return 0;
}
```

新增函数_matmul（用于pybind11注册，注意命名与算子名分开，否则会冲突），函数参数中的tensor对象皆以torch::Tensor类型接收，其他基本类型保持和算子类型一致。

函数首先调度框架函数torchTensor2AsdTensor将torch::Tensor映射为AsdOps::Tensor；

其次从框架函数getCurrentStream获取当前NPU卡对应的stream;

调度自定义算子matMul。

注册过程如下：

```c++
PYBIND11_MODULE(libasdsip, m) {
    m.def("matmul", &_matMul, "function dispatching to matmul operator.");
}
```

### ST框架构建方式介绍

首先导入必要的环境变量（XXXX为当前环境用户）：

```shell
source /home/XXXX/Install_CANN-master/Ascend/set_env.sh
source /home/XXXX/codes/ascend-op-common-lib/output/asdops/set_env.sh
export LD_LIBRARY_PATH=/home/XXXX/codes/AscendSiPBoost/output/lib:/home/XXXX/codes/AscendSiPBoost/build/tests/st:/home/XXXX/.conda/envs/conda-XXXX/lib/python3.9/site-packages/torch/lib:/home/XXXX/.conda/envs/conda-XXXX/lib/python3.9/site-packages/torch_npu/lib:$LD_LIBRARY_PATH
```

在加速库根目录执行构建脚本（注意要加上st参数）：

```shell
bash build.sh st
```

执行成功后在build/tests/st目录生成libasdsip.so链接库文件。

### Python侧算子调用方式介绍

以matMul算子为例：

```python
# 导入st链接库
import libasdsip
# 导入torch
import torch
#导入torch_npu
import torch_npu

# 创建入参tensor（已转成npu对象）
inTensorA = torch.randn((3, 5), dtype=torch.float32).npu()
inTensorB = torch.randn((5, 4), dtype=torch.float32).npu()

# 创建出参tensor（已指定shape和dtype）
outTensor = torch.empty((3, 4), dtype=torch.float32).npu()

# 先设置全局使用的device id（默认0号设备，该步骤非必须）
libasdsip.setDevId(1)

# 调度算子前设置算子使用的device（框架从上下文中取全局device ID做绑定）
libasdsip.setDevice()

# 调度算子
libasdsip.matmul(inTensorA, inTensorB, 3, 5, 4, outTensor)

# 调度算子后重置算子使用的device
libasdsip.resetDevice()

# 借助torch_npu的能力做同步
torch.npu.synchronize()
```

