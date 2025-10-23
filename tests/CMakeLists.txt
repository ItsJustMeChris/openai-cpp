add_library(test_support INTERFACE)
target_include_directories(test_support INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/support
    ${CMAKE_CURRENT_LIST_DIR}/../external
)

target_link_libraries(test_support INTERFACE openai-cpp)

target_compile_features(test_support INTERFACE cxx_std_17)

target_compile_definitions(test_support INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:_SILENCE_ALL_MSVC_DEPRECATION_WARNINGS>
)

add_executable(openai-cpp-tests
    ${CMAKE_CURRENT_LIST_DIR}/client_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utils_uuid_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/multipart_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/responses_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/files_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/streaming_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/chat_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/images_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utils_qs_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utils_env_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utils_to_file_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utils_values_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utils_time_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/batches_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/containers_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/audio_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vector_stores_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/assistants_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/threads_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/messages_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/assistant_stream_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/run_steps_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runs_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/videos_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fine_tuning_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/webhooks_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/conversations_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/moderations_extra_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/azure_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/beta_realtime_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/beta_realtime_transcription_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/beta_chatkit_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/beta_threads_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/uploads_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fine_tuning_graders_test.cpp
)

target_link_libraries(openai-cpp-tests PRIVATE
    test_support
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(openai-cpp-tests)
