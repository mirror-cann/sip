# 信号处理加速库SwapOperation C++ Demo
## 介绍
该目录下为信号加速库SwapOperation C++调用示例。

## 功能说明

- 算子功能：交换两个实（复）数向量
- 计算公式：
    $$
    \begin{aligned}
    \mathbf{x} &\leftrightarrow \mathbf{y} \\
    x_i &\leftrightarrow y_i \quad \text{for } i = 1, 2, \dots, n
    \end{aligned}
    $$


## 使用说明

### 环境配置
 - source 对应的CANN包
    ```
    source [cann安装路径]/set_env.sh
    ```  
    默认：source /usr/local/Ascend/ascend-toolkit/set_env.sh
### SiP编译
 - 进入SiP根目录，执行如下指令进行信号库的编译，设置加速库环境变量。
    ```sh
    cd sip
    bash build.sh
    source output/set_env.sh
    ```
    注意：上述编译方式仅支持编译通过git下载的加速库，以zip压缩包方式下载的加速库暂不支持该编译方式；编译过程需要实时下载依赖库，因此编译环境需要满足联网条件；该编译过程涉及①拉取ascend-boost-comm组件并编译以及②编译加速库两个过程。更多命令介绍可查看SiP仓`build.sh`文件。

 - 更多编译命令说明请参考[编译与构建](../../../../docs/编译与构建.md)

### 运行demo
- 进入 `example` 目录并执行构建脚本。
    ```sh
    cd  ${用例所在目录}
    bash build.sh
    ```
## 额外说明
示例中生成的数据不代表实际场景，可根据具体使用场景进行数据修改。

## 产品支持情况
仅适用于Atlas A2/A3训练系列产品、Atlas 800I A2推理产品、Atlas A3 推理系列产品。

## 场景说明
提供示例代码分别对应不同场景，编译运行时需要对应更改build脚本：
- **example_sswap.cpp**

    【注】：默认编译脚本可编译运行。

    **输入**
    | TensorName |  DataType  | DataFormat |      Shape      |
    | :--------: | :--------: | :--------: | :-------------: |
    |     x      |  float32   |     nd     |       [n]       |
    |     y      |  float32   |     nd     |       [n]       |

    ---

- **example_cswap.cpp**

    【注】：编译脚本内替换 example_sswap.cpp 为 example_cswap.cpp 可编译运行。

    **输入**
    | TensorName |   DataType   | DataFormat |      Shape      |
    | :--------: | :----------: | :--------: | :-------------: |
    |     x      |  complex64   |     nd     |       [n]       |
    |     y      |  complex64   |     nd     |       [n]       |