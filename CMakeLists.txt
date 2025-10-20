cmake_minimum_required(VERSION 3.16)
project(openai_cpp LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CURL REQUIRED)

add_library(openai-cpp
    src/client.cpp
    src/http_client.cpp
    src/utils/base64.cpp
)

target_include_directories(openai-cpp
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/external
)

target_link_libraries(openai-cpp PRIVATE CURL::libcurl)

target_compile_features(openai-cpp PUBLIC cxx_std_17)

add_library(openai::openai ALIAS openai-cpp)

add_executable(openai-cli apps/demo.cpp)

target_link_libraries(openai-cli PRIVATE openai-cpp)

target_include_directories(openai-cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external)
