cmake_minimum_required(VERSION 3.4)
project(civ-pass)

# we need LLVM_HOME in order to automatically set LLVM_DIR
set(ENV{LLVM_DIR} /root/llvm-project/build/lib/cmake/llvm)

find_package(LLVM REQUIRED CONFIG)
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
if (${LLVM_VERSION_MAJOR} VERSION_GREATER_EQUAL 10)
    set(CMAKE_CXX_STANDARD 17)
endif ()

add_subdirectory(TraceDiff)