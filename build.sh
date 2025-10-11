#!/bin/bash
#
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

set -e

function fn_init_use_cxx11_abi()
{
    res=$(python3 -c "import torch" &> /dev/null || echo "torch_not_exist")
    if [ "$res" == "torch_not_exist" ]; then
        echo "Warning: Torch is not installed!"
        [[ "$USE_CXX11_ABI" == "" ]] && USE_CXX11_ABI=ON
        echo "USE_CXX11_ABI=$USE_CXX11_ABI"
        return 0
    fi

    if [ "$USE_CXX11_ABI" == "" ]; then
        if [ "$(python3 -c 'import torch; print(torch.compiled_with_cxx11_abi())')" == "True" ]; then
            USE_CXX11_ABI=ON
        else
            USE_CXX11_ABI=OFF
        fi
    fi
    echo "USE_CXX11_ABI=$USE_CXX11_ABI"
}

function fn_build_mki()
{
    cd $MKI_ROOT
    [[ -d "$CODE_ROOT"/../Mind-KernelInfra/build ]] && rm -rf $CODE_ROOT/../Mind-KernelInfra/build
    [[ -d "$CODE_ROOT"/../Mind-KernelInfra/output ]] && rm -rf $CODE_ROOT/../Mind-KernelInfra/output

    echo  "current commid id of Mind-KernelInfra: $(git rev-parse HEAD)"

    build_options="--use_cxx11_abi=0"
    bash scripts/build.sh $build_options
}

function fn_make_run_package()
{
    if [ $( uname -a | grep -c -i "x86_64" ) -ne 0 ]; then
        echo "it is system of x86_64"
        ARCH="x86_64"
    elif [ $( uname -a | grep -c -i "aarch64" ) -ne 0 ]; then
        echo "it is system of aarch64"
        ARCH="aarch64"
    else
        echo "it is not system of aarch64 or x86_64"
        exit 1
    fi
    branch=$(git symbolic-ref -q --short HEAD || git describe --tags --exact-match 2> /dev/null || echo $branch)
    commit_id=$(git rev-parse HEAD)
    touch $OUTPUT_DIR/version.info
    cat>$OUTPUT_DIR/version.info<<EOF
    Ascend-cann-asdsip : ${VERSION}
    Ascend-cann-asdsip Version : ${VERSION_B}
    Platform : ${ARCH}
    branch : ${branch}
    commit id : ${commit_id}
EOF

    rm -rf $CODE_ROOT/output/host
    rm -rf $CODE_ROOT/output/device/

    # api.h 归档
    chmod 755 -R $CODE_ROOT/scripts/install.sh
    cp $CODE_ROOT/scripts/install.sh $CODE_ROOT/output/
    cp $CODE_ROOT/scripts/set_env.sh $CODE_ROOT/output/


    chmod 755 -R $CODE_ROOT/output
    chmod 755 -R $CODE_ROOT/scripts/makeself

    ARCH=`uname -i`

    mkdir -p $OUTPUT_DIR/scripts
    cp $CODE_ROOT/scripts/install.sh $OUTPUT_DIR
    cp $CODE_ROOT/scripts/set_env.sh $OUTPUT_DIR
    cp $CODE_ROOT/scripts/uninstall.sh $OUTPUT_DIR/scripts
    cp $CODE_ROOT/scripts/filelist.csv $OUTPUT_DIR/scripts
    sed -i "s/ASDSIPPKGARCH/${ARCH}/" $OUTPUT_DIR/install.sh
    sed -i "s!VERSION_PLACEHOLDER!${VERSION}!" $OUTPUT_DIR/install.sh
    sed -i "s!LOG_PATH_PLACEHOLDER!${LOG_PATH}!" $OUTPUT_DIR/install.sh
    sed -i "s!LOG_NAME_PLACEHOLDER!${LOG_NAME}!" $OUTPUT_DIR/install.sh
    sed -i "s!VERSION_PLACEHOLDER!${VERSION}!" $OUTPUT_DIR/scripts/uninstall.sh
    sed -i "s!LOG_PATH_PLACEHOLDER!${LOG_PATH}!" $OUTPUT_DIR/scripts/uninstall.sh
    sed -i "s!LOG_NAME_PLACEHOLDER!${LOG_NAME}!" $OUTPUT_DIR/scripts/uninstall.sh

    $CODE_ROOT/scripts/makeself/makeself.sh --header $CODE_ROOT/scripts/makeself/makeself-header.sh \
        --help-header $CODE_ROOT/scripts/help.info --pigz --complevel 4 --nomd5 --sha256 --chown \
        $CODE_ROOT/output $OUTPUT_DIR/Ascend-cann-SIP_${VERSION}_linux-${ARCH}.run ASCEND_SIP_RUN_PACKAGE ./install.sh

    echo "Ascend-cann-SIP_${VERSION}_linux-${ARCH}.run is successfully generated in $OUTPUT_DIR"
}

function fn_compile_and_pack()
{
    cmake $1 $2
    if [ "$USE_VERBOSE" == "ON" ];then
        VERBOSE=1 make -j
    else
        make -j64
    fi
    make install
    fn_make_run_package
}

function fn_build()
{
    # check dependcy mki
    MKI_PATH="${THIRD_PARTY_DIR}/mki"
    COMPILER_PATH="${THIRD_PARTY_DIR}/compiler"
    if [ ! -d "${MKI_PATH}" ] || [ ! -d "${COMPILER_PATH}" ]; then
        if [ ! -d "${MKI_ROOT}" ]; then
            echo "Third_party dir does not complete and Mind-KernelInfra does not exit!"
            return 0
        fi
        cd $CODE_ROOT/
        mkdir -p 3rdparty
        rm -rf MKI_PATH
        rm -rf COMPILER_PATH
        fn_build_mki
        cp -r $CODE_ROOT/../Mind-KernelInfra/output/mki ${THIRD_PARTY_DIR}/
        cp -r $CODE_ROOT/../Mind-KernelInfra/3rdparty/compiler ${THIRD_PARTY_DIR}/
    fi

    cd $CODE_ROOT/
    echo  "current commid id of ascendSipBoost: $(git rev-parse HEAD)"

    rm -rf build output
    [[ ! -d build ]] && mkdir build
    cd build

    echo "COMPILE_OPTIONS:$COMPILE_OPTIONS"
    COMPILE_OPTIONS="${COMPILE_OPTIONS}  .."

    fn_compile_and_pack "$CODE_ROOT" "$COMPILE_OPTIONS"
}

function help_info() {
    echo "Usage: bash build.sh [type] [options]"
    echo
    echo "type:"
    echo "help                         Displays help message."
    echo "dev                          仅编译算子库, 若type为空，默认为dev."
    echo "clean                        清除缓存和依赖的三方库."
    echo
    echo "options:"
    echo "--output=<dir>               指定编译输出目录，默认为${repo}/output目录."
    echo "--use_cxx11_abi=0            设置-D_GLIBCXX_USE_CXX11_ABI=0 (默认选项)."
    echo "--use_cxx11_abi=1            设置-D_GLIBCXX_USE_CXX11_ABI=1."
    echo "--verbose                    打印详细的编译命令."
    echo "--mssanitizer                启用mssanitizer."
    echo
}

function fn_main()
{
     if [[ "$BUILD_OPTION_LIST" =~ "$1" ]];then
        if [[ -z "$1" ]];then
            arg1="dev"
        else
            arg1=$1
            shift
        fi
    else
        cfg_flag=0
        for item in ${BUILD_CONFIGURE_LIST[*]};do
            if [[ "$1" =~ $item ]];then
                cfg_flag=1
                break 1
            fi
        done
        if [[ "$cfg_flag" == 1 ]];then
            arg1="dev"
        else
            echo "argument $1 is unknown, please type 'build.sh help' for more imformation"
            exit 1
        fi
    fi

    until [[ -z "$1" ]]
    do {
        arg2=$1
        case "${arg2}" in
        --output=*)
            arg2=${arg2#*=}
            if [ -z "$arg2" ];then
                echo "the output directory is not set. This should be set like --output=<outputDir>"
            else
                cd $CURRENT_DIR
                if [ ! -d "$arg2" ];then
                    mkdir -p $arg2
                fi
                export OUTPUT_DIR=$(cd $arg2; pwd)
            fi
            ;;
        "--use_cxx11_abi=1")
            USE_CXX11_ABI=ON
            ;;
        "--use_cxx11_abi=0")
            USE_CXX11_ABI=OFF
            ;;
        "--verbose")
            USE_VERBOSE=ON
            export VERBOSE=1
            ;;
        "--mssanitizer")
            MSSANITIZER_SWITCH=ON
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_MSSANITIZER=OFF"
            ;;
        esac
        shift
    }
    done

    COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_CXX11_ABI=$USE_CXX11_ABI"
    COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=Release"
    case "${arg1}" in
        "dev")
            fn_build
            ;;
        "clean")
            [[ -d "$OUTPUT_DIR" ]] && rm -rf $OUTPUT_DIR
            [[ -d "$THIRD_PARTY_DIR" ]] && rm -rf $THIRD_PARTY_DIR
            echo "clear all build history."
            ;;
        *)
            help_info
            ;;
    esac
}

set -e
cd $(dirname $0)

CURRENT_DIR=$(pwd)
export CODE_ROOT=${CURRENT_DIR}
MKI_ROOT=$CODE_ROOT/../Mind-KernelInfra
export CACHE_DIR=$CODE_ROOT/build
export OUTPUT_DIR=$CODE_ROOT/output
THIRD_PARTY_DIR=$CODE_ROOT/3rdparty
ASDSIP_DIR=$CODE_ROOT
RELEASE_DIR=$CODE_ROOT/ci/release
VERSION="8.2.RC1"
VERSION_B="8.2.RC1"
LOG_PATH="/var/log/cann_asdsip_log/"
LOG_NAME="cann_asdsip_install.log"

export COMPILE_OPTIONS="-DNO_WERROR=ON"
export USE_VERBOSE=OFF
export USE_CXX11_ABI="OFF"
MSSANITIZER_SWITCH=OFF
BUILD_OPTION_LIST="ops_unit ut st ft dev clean help"
BUILD_CONFIGURE_LIST=("--output=.*" "--use_cxx11_abi=0" "--use_cxx11_abi=1 --verbose --mssanitizer")

fn_main "$@"