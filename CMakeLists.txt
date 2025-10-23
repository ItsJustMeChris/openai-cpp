cmake_minimum_required(VERSION 3.16)
project(openai_cpp LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CURL REQUIRED)

option(OPENAI_CPP_BUILD_TESTS "Build openai-cpp tests" OFF)

add_library(openai-cpp
    src/client.cpp
    src/http_client.cpp
    src/utils/base64.cpp
    src/utils/multipart.cpp
    src/utils/env.cpp
    src/utils/time.cpp
    src/utils/uuid.cpp
    src/utils/platform.cpp
    src/utils/qs.cpp
    src/utils/to_file.cpp
    src/utils/values.cpp
    src/logging.cpp
    src/chat.cpp
    src/moderations.cpp
    src/responses.cpp
    src/files.cpp
    src/images.cpp
    src/audio.cpp
    src/containers.cpp
    src/videos.cpp
    src/fine_tuning.cpp
    src/webhooks.cpp
    src/conversations.cpp
    src/beta.cpp
    src/azure.cpp
    src/vector_stores.cpp
    src/batches.cpp
    src/assistants.cpp
    src/threads.cpp
    src/messages.cpp
    src/assistant_stream.cpp
    src/run_steps.cpp
    src/runs.cpp
    src/streaming.cpp
    src/uploads.cpp
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

target_compile_definitions(openai-cpp PUBLIC OPENAI_CPP_VERSION=\"${PROJECT_VERSION}\")

add_library(openai::openai ALIAS openai-cpp)

add_executable(openai-cli apps/demo.cpp)

target_link_libraries(openai-cli PRIVATE openai-cpp)

target_include_directories(openai-cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external)

if (OPENAI_CPP_BUILD_TESTS)
  include(CTest)
  include(FetchContent)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(BUILD_GTEST ON CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(googletest)

  add_subdirectory(tests)
endif()
