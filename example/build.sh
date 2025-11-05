#!/bin/bash

# 检查环境变量 ASDSIP_HOME_PATH 是否已设置
if [ -z "$ASDSIP_HOME_PATH" ]; then
    echo "the env params ASDSIP_HOME_PATH is not set."
    exit 1
fi

# 检查目录是否存在
if [ ! -d "$ASDSIP_HOME_PATH" ]; then
    # 去掉尾部的 'latest'
    new_path="${ASDSIP_HOME_PATH%latest/}"
    # 检查新目录是否存在
    if [ ! -d "$new_path" ]; then
        echo "the env params ASDSIP_HOME_PATH is not exist."
        exit 1
    fi
    export ASDSIP_HOME_PATH=$new_path
    export LD_LIBRARY_PATH=$ASDSIP_HOME_PATH/lib:$LD_LIBRARY_PATH
fi

g++  example.cpp \
    -I${ASCEND_HOME_PATH}/include/aclnn \
    -I${ASCEND_HOME_PATH}/include \
    -L${ASCEND_HOME_PATH}/lib64/ -lascendcl -lopapi -lnnopbase \
    -I${ASDSIP_HOME_PATH}/include \
    -L${ASDSIP_HOME_PATH}/lib -lmki \
    -L${ASDSIP_HOME_PATH}/lib -lasdsip \
    -L${ASDSIP_HOME_PATH}/lib -lasdsip_core \
    -L${ASDSIP_HOME_PATH}/lib -lasdsip_host \
    -o example

echo "Execute the example, calculate the dot product of x and y:"
./example
rm example