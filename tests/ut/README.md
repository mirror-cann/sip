# **AscendSiPBoost 信号处理加速库ST框架介绍**

## 框架目录：

├─tests //测试目录

│  ├─framework //测试框架

|  | ├─test_util // C++ ut测试框架相关目录
|  | | ├─FilterTool.py // 代码过滤工具类，使用lcov工具统计ut代码覆盖率会使用过滤非算子代码
|  | | ├─float_util.cpp/.h // 浮点数比较和数据生成工具类
|  | | ├─golden.cpp/.h  // tensor输入输出比较类
|  | | ├─op_test.cpp/.h  // 测试框架类，与ops_base.cpp/.h类似
|  | | ├─util.cpp/.h  // 测试框架类，与ops_base.cpp/.h类似
|  | ├─torch_extension // ST测试框架类
│  ├─CMakeLists.txt //编译文件

### UT框架构建方式介绍
注：1. XXXX为当前环境用户
注：2. ${WORKSPACE}为用户仓库路径，请根据实际路径修改

1.导入必要的环境编变量

```shell
source /home/XXXX/Install_CANN-master/Ascend/set_env.sh
```

2.编译ascend-op-common-lib仓库：

```shell
cd ${WORKSPACE}/ascend-op-common-lib/
rm -rf output build
bash scripts/build.sh dev --use_cxx11_abi=0
```

3.导入必要的环境编变量

```shell
source  ${WORKSPACE}/ascend-op-common-lib/output/asdops/set_env.sh
export LD_LIBRARY_PATH=/home/xxx/.conda/envs/conda-xxx/lib/python3.9/site-packages/torch.libs:$LD_LIBRARY_PATH
export LD_PRELOAD=/home/xxx/.conda/envs/conda-xxx/lib/libgomp.so.1:$LD_PRELOAD
```

4.在加速库根目录执行构建脚本（注意要加上st参数）：

```shell
cd ${WORKSPACE}/AscendSiPBoost
bash build.sh --use_cxx11_abi=0 ut 
```

### UT用例编写相关接口介绍
以transpose为例，
1.测试kernel
编写流程与example的main函数调用算子流程基本一样。

（1）创建tensor描述信息数据结构 inTensorDesc
```C++
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 2}});
```
（2）创建算子对应的oParm属性信息，并生成算子描述信息OpDesc
```C++
    OpParam::Transpose opParam = {};
    opParam.perm = {0, 1};
    OpDesc opDesc = {0, "TransposeOperation", opParam};
```
（3）初始化，即设置NPU的deviceid（默认0）并创建stream
```C++
    int deviceId = 0;
    AsdRtStream stream = OpTestInit(deviceId);
```
（4）创建tensor
```C++
    OpTestMallocInTensors(inTensorDescs, context);
```
 (5) 调用接口计算
```C++
    Status status = transpose(tensorIn, tensorOut, 1, 0, stream);
```
 (6) 将tensorOut的device侧值搬运到host侧
```C++
    CopyDeviceTensorToHostTensor(tensorOut);
```
 (7) 创建TensorContext，记录输入输出值以及算子信息
```C++
    TensorContext context;
    context.opDesc = opDesc;
    context.outTensors.push_back(tensorOut);
```
 (8) 调用比较函数，生成标杆值并进行比较。
```C++
    status = TransposeCompare(0.001, 0.001, context);
```
 (9) 内存释放
```C++
    OpTestEnd(deviceId, context, stream);
```
 (10) 根据需要进行断言
```C++
    ASSERT_EQ(status.Ok(), true);
```

2.测试tiling文件
(1) 创建LaunchParam类，记录算子输入输出以及opParam属性信息
```C++
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1, 5}}});
    OpParam::Transpose opParam = {};
    opParam.perm = {1, 0};
    OpDesc opDesc = {0, "TransposeOperation", opParam};
    launchParam.SetOpDesc(opDesc);
```
(2) 实例化算子operation类，并推断output的shape，并记录于LaunchParam类中
```C++
    AsdOps::Operation *op = AsdOps::Ops::Instance().GetOperationByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
```
(3) 获取算子的tactic类，获取tiling数据所需要的空间大小，并记录于LaunchParam类中
```C++
    sdOps::Tactic *tactic = op->GetTacticByName("TransposeF32Tactic");
    uint32_t launchBufferSize = tactic->GetTilingSize(launchParam);
```
(4) 为tiling数据指针开辟空间
```C++
    runInfo.GetKernelInfo().AllocTilingHost(launchBufferSize);
```
(5) 调用算子tiling方法
```C++
    status = TransposeTiling("TransposeF32Tactic", launchParam, runInfo);
```

3.compare函数相关接口介绍
（1）inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    const std::function<void(void*)>& deleter,
    const TensorOptions& options = {},
    const c10::optional<Device> target_device = c10::nullopt)
讲tensor转换成pytorch中的tensor, 其中data是内存上指针（host侧）, sizes指张量的形状，strides指张量步幅，其他为可选参数，暂不关注。
```C++
atInTensor = at::from_blob(inTensor.hostData, ToIntArrayRef(inTensor.desc.dims), at::kFloat);
```
（2）bool allclose(const Tensor& self, const Tensor& other, double rtol = 1e-5, double atol = 1e-8, bool equal_nan = false);
比较两个张量是否在容差范围内相等，其中rtol是相对误差，atol是绝对误差，equal_nan指定是否将 NaN 视为相等，默认值为 `false`
```C++
at::allclose(atOutRefTensor, atOutTensor, EXTENT_OF_ERROR, EXTENT_OF_ERROR)
```
