set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
set(CMAKE_SKIP_RPATH TRUE)

string(REGEX REPLACE "[^A-Za-z0-9_]" "" NAMESPACE "${NAMESPACE}")

if(NOT DEFINED NO_WERROR)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>")
endif()

add_compile_options(
    -fexceptions
    "$<$<COMPILE_LANGUAGE:CXX>:-std=c++17;-pipe;-Wno-unused-parameter;-Wno-ignored-qualifiers>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Wformat=0;-fno-strict-aliasing>"
    "$<$<COMPILE_LANGUAGE:CXX>:-fPIC;-fstack-protector-all;-Wl,--build-id=none>"
)

if(NAMESPACE STREQUAL "")
    set(NAMESPACE "Mki")
endif()

message(STATUS "The namespace for infra would be: ${NAMESPACE}")

add_compile_definitions(
    OpSpace=${NAMESPACE}
    "$<$<COMPILE_LANGUAGE:CXX>:_GLIBCXX_USE_CXX11_ABI=$<BOOL:${USE_CXX11_ABI}>>"
)

set(LD_FLAGS_GLOBAL "-shared;-rdynamic;-ldl;-Wl,-z,relro;-Wl,-z,now")
set(LD_FLAGS_GLOBAL "${LD_FLAGS_GLOBAL};-Wl,-z,noexecstack;-Wl,--build-id=none")

add_link_options(
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${LD_FLAGS_GLOBAL};-fexceptions>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${LD_FLAGS_GLOBAL};-pie;-fPIE>"
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    ADD_LINK_OPTIONS(-s)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_DEBUG -g")
endif()

if(ENABLE_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -O3")
    set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)
endif()
