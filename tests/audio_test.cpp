#include <gtest/gtest.h>

#include "openai/client.hpp"
#include "openai/audio.hpp"
#include "support/mock_http_client.hpp"

#include <filesystem>
#include <fstream>

namespace oait = openai::testing;

TEST(AudioTranscriptionsResourceTest, CreateParsesResponse) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string response_body = R"({"text":"hello world","usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}})";
  mock_ptr->enqueue_response(HttpResponse{200, {}, response_body});

  std::filesystem::path tmp = std::filesystem::temp_directory_path() / "openai-cpp-audio.wav";
  {
    std::ofstream out(tmp, std::ios::binary);
    out << "audio";
  }

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  TranscriptionRequest request;
  request.file.purpose = "assistants";
  request.file.file_path = tmp.string();
  request.file.file_name = "audio.wav";
  request.file.content_type = "audio/wav";
  request.model = "whisper-1";

  auto transcription = client.audio().transcriptions().create(request);
  EXPECT_EQ(transcription.text, "hello world");
  ASSERT_TRUE(transcription.usage.has_value());
  EXPECT_EQ(transcription.usage->total_tokens, 15);

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& req = *mock_ptr->last_request();
  EXPECT_NE(req.headers.at("Content-Type").find("multipart/form-data"), std::string::npos);
  EXPECT_NE(req.body.find("audio"), std::string::npos);

  std::filesystem::remove(tmp);
}

