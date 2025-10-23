#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "openai/beta.hpp"
#include "openai/client.hpp"
#include "support/mock_http_client.hpp"

namespace openai {
namespace {

using namespace openai::testing;

TEST(BetaRealtimeTranscriptionSessionsTest, CreateSendsBetaHeaderAndBody) {
  auto http = std::make_unique<MockHttpClient>();
  auto* mock_ptr = http.get();

  const std::string response_body = R"({
    "client_secret": {
      "expires_at": 123456,
      "value": "temporary"
    },
    "input_audio_format": "pcm16",
    "input_audio_transcription": {"language": "en", "model": "gpt-4o-mini-transcribe"},
    "modalities": ["text"],
    "turn_detection": {"type": "server_vad", "prefix_padding_ms": 150}
  })";
  mock_ptr->enqueue_response(HttpResponse{200, {}, response_body});

  ClientOptions options;
  options.api_key = "sk-test";
  OpenAIClient client(std::move(options), std::move(http));

  beta::RealtimeTranscriptionSessionCreateParams params;
  params.include = std::vector<std::string>{"item.input_audio_transcription.logprobs"};
  params.input_audio_format = "pcm16";
  beta::RealtimeTranscriptionSessionCreateInputAudioNoiseReduction nr;
  nr.type = "near_field";
  params.input_audio_noise_reduction = nr;
  beta::RealtimeTranscriptionSessionCreateInputAudioTranscription transcription;
  transcription.model = "gpt-4o-mini-transcribe";
  transcription.language = "en";
  params.input_audio_transcription = transcription;
  beta::RealtimeTranscriptionSessionCreateTurnDetection detection;
  detection.type = "server_vad";
  detection.prefix_padding_ms = 150;
  params.turn_detection = detection;

  auto session = client.beta().realtime().transcription_sessions().create(params);
  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& request = *mock_ptr->last_request();
  EXPECT_EQ(request.method, "POST");
  EXPECT_NE(request.url.find("/realtime/transcription_sessions"), std::string::npos);
  EXPECT_EQ(request.headers.at("OpenAI-Beta"), "assistants=v2");

  auto payload = nlohmann::json::parse(request.body);
  EXPECT_EQ(payload.at("input_audio_format"), "pcm16");
  EXPECT_EQ(payload.at("include")[0], "item.input_audio_transcription.logprobs");
  EXPECT_EQ(payload.at("input_audio_noise_reduction").at("type"), "near_field");
  EXPECT_EQ(payload.at("turn_detection").at("type"), "server_vad");

  ASSERT_TRUE(session.input_audio_format.has_value());
  EXPECT_EQ(*session.input_audio_format, "pcm16");
  ASSERT_TRUE(session.client_secret.has_value());
  EXPECT_EQ(session.client_secret->value, "temporary");
}

}  // namespace
}  // namespace openai
