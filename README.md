# AscendSiPBoost 信号处理加速库

Ascend Signal Processing Boost 昇腾信号处理加速库（下文简称为SIP库）是一款高效、可靠的加速库，基于华为Ascend AI处理器，专门为信号处理领域设计的高性能算子加速库。

## 内容总览
1. [学习资源](#学习资源)
2. [什么是SIP](#什么是SIP)
3. [环境构建](#环境构建)
4. [快速上手](#快速上手)
5. [自定义算子开发](#自定义算子开发)
6. [参与贡献](#参与贡献)
7. [参考文档](#参考文档)
   
## 学习资源
 
- [编译与构建](docs/编译与构建.md)：SIP的编译命令说明。
- [API文档](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/SiPAPI/SIP_API_0002.html)：介绍了SIP库的接口和相关术语。
- [问题报告](https://gitcode.com/cann/sip/issues)：通过issue提交发现的问题。

## 什么是SIP
### SIP介绍
Ascend Signal Processing Boost 昇腾信号处理加速库（下文简称为SIP库）是一款高效、可靠的加速库，基于华为Ascend AI处理器，专门为信号处理领域设计的高性能算子加速库。
### 软件架构
加速库接口功能主要分成六个部分：
- 信号处理加速库框架：负责算子的管理，算子在Device侧的二进制加载及host侧的tiling；负责对上层提供接口支持单算子调用、多算子批量调用等。
- FFT库：包括专用的NPU Kernel、PLAN框架；对标cufft库，对外提供接口以实现C2C、C2R和R2C，供开发者使用。
- BLAS库：依照BLAS相关的标准定义，提供专用的Kernel，对标cuBLAS库，对外提供从level1到level3的接口，供开发者使用。
- 复数基础计算库：提供基础的支持复数类型的算子。
- 信号领域融合算子库：包含PC、MTD、CFAR、Interpolation等融合算子，支撑脉冲信号分析，动态目标检测，恒虚警等场景。
- Solver库：主要提供基于BLAS的复杂线性代数函数，例如矩阵分解、特征值求解等。

### SIP仓介绍
SIP库的目录结构如下：

```
AscendSiPBoost
├── 3rdparty            //第三方依赖库文件夹
├── build               //可存放构建生成的文件
├── docs                //文档文件
├── example             //算子调用示例代码，包含可直接运行的Demo
├── include             //存放公共头文件
├── configs             //存放算子输入输出数据规格约束文件
├── output              //编译输出文件夹
├── scripts             //脚本文件存放目录
├── core                //SIP库算子对外调用接口目录
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
│   ├── interpolation
│   ├── include
│   ├── utils
│   ├── ops.cpp
│   └── CMakeLists.txt
├── build.sh            //构建脚本
└── CMakeLists.txt      //构建脚本
```

### 为什么选择SIP
- 支持FFT、BLAS、FIR滤波、插值等高性能NPU算子，充分利用Ascend AI处理器的硬件特性，包括硬件算力、存储带宽、内存带宽等，实现了算子的极致性能。

## 环境构建
### 快速安装CANN软件
本节提供快速安装CANN软件的示例命令，更多安装步骤请参考[详细安装指南](#详细安装指南)。

#### 安装前准备
在线安装和离线安装时，需确保已具备Python环境及pip3，当前CANN支持Python3.7.x至3.11.4版本。
离线安装时，请单击[获取链接](https://www.hiascend.com/developer/download/commercial/result?module=cann)下载CANN软件包，并上传到安装环境任意路径。
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
开发者可访问[昇腾文档-昇腾社区](https://www.hiascend.com/document)->CANN商用版->软件安装，查看CANN软件安装引导，根据机器环境、操作系统和业务场景选择后阅读详细安装步骤。

### 基础工具版本要求与安装

安装CANN之后，还需要安装一些工具方便后续开发，参见以下内容：

* [CANN依赖列表](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0045.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)
* [CANN安装后操作](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0094.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)

## 快速上手
### SIP编译
 - 加速库下载
    ```sh
    git clone https://gitcode.com/cann/sip.git
    ```
   您可自行选择需要的分支。
 - SIP库编译<br>
    编译加速库，设置加速库环境变量：
    ```sh
    cd AscendSiPBoost
    bash build.sh
    source output/SIP/set_env.sh
    ```
    注意：该编译过程涉及①拉取算子库/MKI并编译②加速库的编译两个过程。更多命令介绍可查看SIP仓`build.sh`文件。
 - 无法获取SIP代码仓时，可通过安装nnal软件包获取对应so文件<br>
    - 安装步骤可参考 `run包使用`
    - 代码及软件包版本对应关系：<br>
        nnal软件包需保持和toolkit及kernels软件包版本一致
        |CANN|代码分支|
        |-|-|
        |CANN 8.1.RC1|br_feature_cann_8.2.RC1_0515POC_20250630|

    - 执行 
        ```sh
        source {install path}/nnal/asdsip/set_env.sh
        ```
 - run包使用<br>
    - run包获取
    1. 进入网址：https://www.hiascend.com/developer/download/commercial
    2. 产品系列选择服务器，产品型号根据设备型号选择，选择所需解决方案版本，随后在CANN区域选择软件包跟随指引即可获取相关run包
    - 软件包名为：Ascend-cann-SIP_{version}_linux-{arch}.run <br>
    其中，{version}表示软件版本号，{arch}表示CPU架构。
    - 安装run包（需要依赖cann环境）
        ```sh
        chmod +x 软件包名.run # 增加对软件包的可执行权限
        ./软件包名.run --check # 校验软件包安装文件的一致性和完整性
        ./软件包名.run --install # 安装软件，可使用--help查询相关安装选项
        ```
        出现提示`xxx install success!`则安装成功

 - 更多编译命令说明请参考[编译与构建](docs/编译与构建.md)
### 调用示例说明
本节示例代码分别展示了如何通过C++调用算子。

#### C++

在SIP仓库的`example`目录下，存放了多个不依赖测试框架、即编可用的算子调用Demo示例。进入对应目录执行如下命令就可完成一个算子的调用执行。代码完整内容可参考`example\A2\example_sdot.cpp`，下面仅展示其核心内容：
```c++
#include <iostream>
#include <vector>
#include "asdsip.h"
#include "acl/acl.h"
#include "acl_meta.h"

using namespace AsdSip;

#define ASD_STATUS_CHECK(err)                                                \
    do {                                                                     \
        AsdSip::AspbStatus err_ = (err);                                     \
        if (err_ != AsdSip::NO_ERROR) {                                      \
            std::cout << "Execute failed." << std::endl; \
            exit(-1);                                                        \
        }                                                                    \
    } while (0)

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

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
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

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
    int deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

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

    std::vector<int64_t> xShape = {xSize};
    std::vector<int64_t> yShape = {ySize};
    std::vector<int64_t> resultShape = {resultSize};
    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor *result = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *resultDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_FLOAT, &inputX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInYData, yShape, &inputYDeviceAddr, aclDataType::ACL_FLOAT, &inputY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(resultData, resultShape, &resultDeviceAddr, aclDataType::ACL_FLOAT, &result);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeDotPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasSdot(handle, n, inputX, incx, inputY, incy, result));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(resultData.data(),
        resultSize * sizeof(float),
        resultDeviceAddr,
        resultSize * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- result -------" << std::endl;
    for (int64_t i = 0; i < 1; i++) {
        std::cout << resultData[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Execute successfully." << std::endl;

    aclDestroyTensor(inputX);
    aclDestroyTensor(inputY);
    aclDestroyTensor(result);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);
    aclrtFree(resultDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
算子使用指导：可访问[头文件列表-CANN商用版8.2.RC1-昇腾社区](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/SiPAPI/SIP_API_0002.html)。

#### 样例安全声明
`example`目录下的样例旨在提供快速上手、开发和调试SIP特性的最小化实现，其核心目标是使用最精简的代码展示SIP核心功能，**而非提供生产级的安全保障**。与成熟的生产级使用方法相比，此样例中的安全功能（如输入校验、边界校验）相对有限。

SIP不推荐用户直接将样例作为业务代码，也不保证此种做法的安全性。若用户将`example`中的示例代码应用在自身的真是业务场景中且发生了安全问题，则由用户自行承担。

### 日志和环境变量说明
- 加速库日志现在已经部分适配CANN日志，环境变量说明请参考
  **[CANN商用版文档/环境变量参考](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/acce/ascendtb/ascendtb_0032.html)**。


## 自定义算子开发
敬请期待。

## 参与贡献
 
敬请期待。


## 参考文档
**[CANN商用版文档](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/index/index.html)**  
**[SIP商用版文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/acce/SiP/SIP_0000.html)**