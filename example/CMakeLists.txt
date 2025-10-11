# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# file(GLOB_RECURSE SOURCE_FILES example_acl_fft_c2c_1d.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_acl_fft_r2c_1d.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_acl_fft_c2r_1d.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_acl_fft_c2c_2d.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_acl_fft_r2c_2d.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_acl_fft_c2r_2d.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_csrot.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_ssyr.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_scnrm2.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_snrm2.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_cgerc.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_icamax.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_isamax.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_strmv.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_ctrmv.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_ccopy.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_scopy.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_strmm.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_cscal.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_csscal.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_sscal.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_cswap.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_sswap.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_cdotc.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_cdotu.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_sdot.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_cgemm.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_cgemv.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_caxpy.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_ssyr2.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_complex_mat_dot.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_colwise_mul.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_sasum.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_scasum.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_hcgemv_batched.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_cgemv_batched.cpp)
# set(SOURCE_FILES A2/BASE/example_swaplast2axes.cpp)
# set(SOURCE_FILES A2/BASE/example_conj.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_complex32_mul.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_complex64_mul.cpp)
# file(GLOB_RECURSE SOURCE_FILES aclnnAdd_transpose.cpp)
# file(GLOB_RECURSE SOURCE_FILES transpose_aclnnAdd.cpp)
# file(GLOB_RECURSE SOURCE_FILES aclnnforeachSin.cpp)
# file(GLOB_RECURSE SOURCE_FILES transpose_aclnnCumprod.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_rsInterpolationBySinc.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_asd_convolve_complex64.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_asd_convolve_complex32.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_interpWithCoeff_complex64.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_hcgemm_batched.cpp)
# file(GLOB_RECURSE SOURCE_FILES example_cgemm_batched.cpp)
file(GLOB_RECURSE SOURCE_FILES example.cpp)

add_executable(Sample ${SOURCE_FILES})
# target_link_libraries(Sample PUBLIC asdsip asdops asdops_host)
target_link_libraries(Sample PUBLIC asdsip asdsip_core asdsip_host)

# link_directories(
#     $ENV{ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib  # 自定义算子lib路径
# )
# include_directories(
#     $ENV{ASCEND_HOME_PATH}/opp/vendors/customize/op_api/include # 自定义算子头文件路径
# )

include_directories(
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include/aclnn
)

target_link_libraries(Sample PUBLIC asdsip asdsip_core asdsip_host
    $ENV{ASCEND_HOME_PATH}/lib64/libascendcl.so
    $ENV{ASCEND_HOME_PATH}/lib64/libnnopbase.so
    $ENV{ASCEND_HOME_PATH}/lib64/libopapi.so
)

install(TARGETS Sample DESTINATION bin)
