# AscendSiPBoost 信号处理加速库

🔥 [2025/10] AscendSiPBoost项目（下文简称为SiP库）首次上线。

## 内容总览
1. [学习资源](#学习资源)
2. [什么是SiP](#什么是SiP)
3. [环境构建](#环境构建)
4. [快速上手](#快速上手)
5. [自定义算子开发](#自定义算子开发)
6. [参与贡献](#参与贡献)
7. [参考文档](#参考文档)
   
## 学习资源
 
- [编译与构建](docs/编译与构建.md)：SiP的编译命令说明。
- [API文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/API/SiPAPI/SIP_API_0002.html)：介绍了SiP库的接口和相关术语。
- [问题报告](https://gitcode.com/cann/sip/issues)：通过issue提交发现的问题。

## 什么是SiP
### SiP介绍
Ascend Signal Processing Boost 昇腾信号处理加速库（下文简称为SiP库）是一款高效、可靠的加速库，基于华为Ascend AI处理器，专门为信号处理领域设计的高性能算子加速库。
### 软件架构
加速库接口功能主要分成六个部分：
- 信号处理加速库框架：负责算子的管理，算子在Device侧的二进制加载以及Host侧的tiling；负责对上层提供接口支持单算子调用、多算子批量调用等。
- FFT库：包括专用的NPU Kernel、PLAN框架；实现FFT系列算子功能，对外提供接口以实现C2C、C2R和R2C，供开发者使用。
- BLAS库：依照BLAS相关的标准定义，提供专用的Kernel，实现BLAS系列算子功能，对外提供从level1到level3的接口，供开发者使用。
- 复数基础计算库：提供基础的支持复数类型的算子。
- 信号领域融合算子库：包含PC、MTD、CFAR、Interpolation等融合算子，支撑脉冲信号分析，动态目标检测，恒虚警等场景。
- Solver库：主要提供基于BLAS的复杂线性代数函数，例如矩阵分解、特征值求解等。

### SiP仓介绍
SiP库的目录结构如下：

```
sip
├── docs                //文档文件
├── example             //算子调用示例代码，包含可直接运行的Demo
├── include             //存放公共头文件
├── configs             //存放算子输入输出数据规格约束文件
├── scripts             //脚本文件存放目录
├── core                //SiP库算子对外调用接口目录
│   ├── base
│   ├── blas
│   ├── fft
│   ├── filter
│   ├── interpolation
│   ├── include
│   ├── utils
│   └── CMakeLists.txt
├── imported_libs       //SiP库引入lib相关构建脚本
│   ├── base
│   ├── blas
│   ├── fft
│   ├── filter
│   ├── interpolation
│   ├── include
│   ├── utils
│   └── CMakeLists.txt
├── ops                 //算子存放目录
│   ├── base
│   ├── blas
│   ├── fft
│   ├── filter
│   ├── include
│   ├── utils
│   ├── ops.cpp
│   └── CMakeLists.txt
├── cmake               //SiP库构建相关脚本
├── tests               //SiP库测试相关目录
│   ├── ut
│   └── CMakeLists.txt
├── build.sh 
└── CMakeLists.txt
├── LICENSE
├── OAT.xml
├── OWNERS
├── README.md
|── SECURITY.md
|── Third_Party_Open_Source_Software_Notice
└── version.info
```

### 为什么选择SiP
- 支持FFT、BLAS、FIR滤波、插值等高性能NPU算子，充分利用Ascend AI处理器的硬件特性，包括硬件算力、存储带宽、内存带宽等，实现了算子的极致性能。

## 环境构建
### 快速安装CANN软件
本节提供快速安装CANN软件的示例命令，更多安装步骤请参考[详细安装指南](#详细安装指南)。

#### 安装前准备
在线安装和离线安装时，需确保已具备Python环境及pip3，当前CANN支持Python3.7.x至3.11.4版本。
离线安装时，请单击[获取链接](https://www.hiascend.com/developer/download/community/result?module=cann)下载CANN软件包，并上传到安装环境任意路径。
#### 安装CANN
```shell
chmod +x Ascend-cann-toolkit_8.2.RC1_linux-$(arch).run
./Ascend-cann-toolkit_8.2.RC1_linux-$(arch).run --install
```
#### 安装后配置
配置环境变量脚本set_env.sh，当前安装路径以${HOME}/Ascend为例。
```
source ${HOME}/Ascend/ascend-toolkit/set_env.sh
```  
安装业务运行时依赖的Python第三方库（如果使用root用户安装，请将命令中的--user删除）。
```
pip3 install attrs cython 'numpy>=1.19.2,<=1.24.0' decorator sympy cffi pyyaml pathlib2 psutil protobuf==3.20.0 scipy requests absl-py --user
```
### 详细安装指南 
开发者可访问[昇腾文档-昇腾社区](https://www.hiascend.com/document)->CANN社区版->软件安装，查看CANN软件安装引导，根据机器环境、操作系统和业务场景选择后阅读详细安装步骤。

### 基础工具版本要求与安装

安装CANN之后，还需要安装一些工具方便后续开发，参见以下内容：

* [CANN依赖列表](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/softwareinst/instg/instg_0045.html?Mode=PmIns&OS=Debian&Software=cannToolKit)
* [CANN安装后操作](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/softwareinst/instg/instg_0094.html?Mode=PmIns&OS=Debian&Software=cannToolKit)

## 快速上手
### SiP编译
 - 加速库下载
    ```sh
    git clone https://gitcode.com/cann/sip.git
    ```
   您可自行选择需要的分支。
 - SiP库编译<br>
    编译加速库，设置加速库环境变量：
    ```sh
    cd sip
    bash build.sh
    source output/set_env.sh
    ```
    注意：上述编译方式仅支持编译通过git下载的加速库，以zip压缩包方式下载的加速库暂不支持该编译方式；编译过程需要实时下载依赖库，因此编译环境需要满足联网条件；该编译过程涉及①拉取ascend-boost-comm组件并编译以及②编译加速库两个过程。更多命令介绍可查看SiP仓`build.sh`文件。

 - 更多编译命令说明请参考[编译与构建](docs/编译与构建.md)
### 调用示例说明
本节示例代码分别展示了如何通过C++调用算子。

#### C++

在SiP仓库的`example`目录下，存放了多个不依赖测试框架、即编可用的算子调用Demo示例。本节示例代码展示通过C++调用SiP asdBlasSdot算子实现向量点乘（内积）功能，代码完整内容可参考`example\example.cpp`，下面仅展示其核心内容：
```c++
#include <iostream>
#include <vector>
#include "asdsip.h"
#include "acl/acl.h"
#include "acl_meta.h"

using namespace AsdSip;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，acl初始化
    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(stream);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请Device侧内存
    aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    // 调用aclrtMemcpy将Host侧数据复制到Device侧内存上
    aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        *deviceAddr);
    return 0;
}

int main(int argc, char **argv)
{
    // 设置算子使用的device id
    int deviceId = 0;
    //（固定写法）创造执行流
    aclrtStream stream;
    Init(deviceId, &stream);

    // 创造tensor的Host侧数据
    int64_t n = 5;
    int64_t incx = 1;
    int64_t incy = 1;

    int64_t xSize = 5;
    std::vector<float> tensorInXData;
    tensorInXData.reserve(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        tensorInXData[i] = 1.0 + i;
    }

    int64_t ySize = 5;
    std::vector<float> tensorInYData;
    tensorInYData.reserve(xSize);
    for (int64_t i = 0; i < ySize; i++) {
        tensorInYData[i] = 10.0 + i;
    }

    int64_t resultSize = 1;
    std::vector<float> resultData;
    resultData.reserve(resultSize);

    std::cout << "------- input x -------" << std::endl;
    for (int64_t i = 0; i < xSize; i++) {
        std::cout << tensorInXData[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "------- input y -------" << std::endl;
    for (int64_t i = 0; i < ySize; i++) {
        std::cout << tensorInYData[i] << " ";
    }
    std::cout << std::endl;

    // 创造输入/输出tensor
    std::vector<int64_t> xShape = {xSize};
    std::vector<int64_t> yShape = {ySize};
    std::vector<int64_t> resultShape = {resultSize};
    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor *result = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *resultDeviceAddr = nullptr;
    CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_FLOAT, &inputX);
    CreateAclTensor(tensorInYData, yShape, &inputYDeviceAddr, aclDataType::ACL_FLOAT, &inputY);
    CreateAclTensor(resultData, resultShape, &resultDeviceAddr, aclDataType::ACL_FLOAT, &result);

    // 创建算子执行句柄
    asdBlasHandle handle;
    asdBlasCreate(handle);

    // 创造算子执行所需workspace
    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeDotPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    if (lwork > 0) {
        aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
    }
    asdBlasSetWorkspace(handle, buffer);

    // 配置算子执行信息
    asdBlasSetStream(handle, stream);

    // 调用接口执行算子（固定调用逻辑）
    asdBlasSdot(handle, n, inputX, incx, inputY, incy, result);
    asdBlasSynchronize(handle);

    // 调度算子后销毁算子句柄
    asdBlasDestroy(handle);

    // 将输出tensor的Device侧数据复制到Host侧内存上
    aclrtMemcpy(resultData.data(),
        resultSize * sizeof(float),
        resultDeviceAddr,
        resultSize * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);

    std::cout << "------- result -------" << std::endl;
    for (int64_t i = 0; i < 1; i++) {
        std::cout << resultData[i] << " ";
    }
    std::cout << std::endl;

    // 资源释放
    aclDestroyTensor(inputX);
    aclDestroyTensor(inputY);
    aclDestroyTensor(result);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);
    aclrtFree(resultDeviceAddr);
    if (lwork > 0) {
        aclrtFree(buffer);
    }

    // 调度算子后重置算子使用的deviceId
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
文件编译说明：进入example目录，执行bash build.sh完成编译和执行。
```shell
cd example
bash build.sh
```
算子使用指导：可访问[头文件列表-CANN社区版-昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/API/SiPAPI/SIP_API_0002.html)。

#### 样例安全声明
`example`目录下的样例旨在提供快速上手、开发和调试SiP特性的最小化实现，其核心目标是使用最精简的代码展示SiP核心功能，**而非提供生产级的安全保障**。与成熟的生产级使用方法相比，此样例中的安全功能（如输入校验、边界校验）相对有限。

SiP不推荐用户直接将样例作为业务代码，也不保证此种做法的安全性。若用户将`example`中的示例代码应用在自身的真实业务场景中且发生了安全问题，则由用户自行承担。

### 日志和环境变量说明
- 加速库日志现在已经部分适配CANN日志，环境变量说明请参考
  **[CANN社区版文档/环境变量参考](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/acce/SIP/SIP_0008.html)**。


## 自定义算子开发
详细步骤可参考[从开发一个简单算子出发](docs/从开发一个简单算子出发.md)

## 参与贡献
 
1.  fork仓库
2.  修改并提交代码
3.  新建 Pull-Request

详细步骤可参考[贡献指南](docs/贡献指南.md)


## 参考文档
**[CANN社区版文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/index/index.html)**
**[SiP社区版文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/acce/SiP/SIP_0000.html)**