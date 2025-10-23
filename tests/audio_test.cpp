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

  const std::string response_body =
      R"({"text":"hello world","usage":{"type":"tokens","input_tokens":10,"output_tokens":5,"total_tokens":15,"input_token_details":{"audio_tokens":7,"text_tokens":3}},"logprobs":[{"token":"hello","logprob":-0.1,"bytes":[104,101]},{"token":"world","logprob":-0.2}],"segments":[{"id":0,"avg_logprob":-0.1,"compression_ratio":0.5,"end":1.2,"no_speech_prob":0.01,"seek":0,"start":0.0,"temperature":0.0,"text":"hello world","tokens":[42,43]}],"words":[{"start":0.0,"end":0.5,"word":"hello"},{"start":0.5,"end":1.0,"word":"world"}],"duration":1.23,"language":"english","task":"transcribe"})";
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
  TranscriptionChunkingStrategy strategy;
  strategy.type = TranscriptionChunkingStrategy::Type::ServerVad;
  strategy.prefix_padding_ms = 250;
  strategy.silence_duration_ms = 500;
  strategy.threshold = 0.6;
  request.chunking_strategy = strategy;
  request.include = std::vector<std::string>{"logprobs"};
  request.known_speaker_names = std::vector<std::string>{"agent"};
  request.known_speaker_references = std::vector<std::string>{"data:audio/wav;base64,AAAA"};
  request.language = "en";
  request.prompt = "guide";
  request.response_format = "json";
  request.stream = false;
  request.temperature = 0.5;
  request.timestamp_granularities = std::vector<std::string>{"word", "segment"};

  auto transcription = client.audio().transcriptions().create(request);
  EXPECT_EQ(transcription.text, "hello world");
  ASSERT_TRUE(transcription.usage.has_value());
  EXPECT_EQ(transcription.usage->total_tokens, 15);
  EXPECT_EQ(transcription.usage->type, TranscriptionUsage::Type::Tokens);
  ASSERT_TRUE(transcription.usage->input_token_details.has_value());
  EXPECT_EQ(transcription.usage->input_token_details->audio_tokens.value_or(-1), 7);
  ASSERT_TRUE(transcription.logprobs.has_value());
  EXPECT_EQ(transcription.logprobs->size(), 2u);
  ASSERT_TRUE(transcription.segments.has_value());
  EXPECT_EQ(transcription.segments->at(0).text, "hello world");
  ASSERT_TRUE(transcription.words.has_value());
  EXPECT_EQ(transcription.words->at(1).word, "world");
  ASSERT_TRUE(transcription.duration.has_value());
  EXPECT_DOUBLE_EQ(transcription.duration.value(), 1.23);
  ASSERT_TRUE(transcription.language.has_value());
  EXPECT_EQ(*transcription.language, "english");
  EXPECT_TRUE(transcription.is_verbose);
  EXPECT_FALSE(transcription.is_plain_text);

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& req = *mock_ptr->last_request();
  EXPECT_NE(req.headers.at("Content-Type").find("multipart/form-data"), std::string::npos);
  EXPECT_NE(req.body.find("audio"), std::string::npos);
  EXPECT_NE(req.body.find("chunking_strategy[type]"), std::string::npos);
  EXPECT_NE(req.body.find("chunking_strategy[prefix_padding_ms]"), std::string::npos);
  EXPECT_NE(req.body.find("chunking_strategy[silence_duration_ms]"), std::string::npos);
  EXPECT_NE(req.body.find("chunking_strategy[threshold]"), std::string::npos);
  EXPECT_NE(req.body.find("name=\"include[]\""), std::string::npos);
  EXPECT_NE(req.body.find("\r\n\r\nlogprobs"), std::string::npos);
  EXPECT_NE(req.body.find("name=\"known_speaker_names[]\""), std::string::npos);
  EXPECT_NE(req.body.find("\r\n\r\nagent"), std::string::npos);
  EXPECT_NE(req.body.find("name=\"known_speaker_references[]\""), std::string::npos);
  EXPECT_NE(req.body.find("\r\n\r\ndata:audio/wav;base64,AAAA"), std::string::npos);
  EXPECT_NE(req.body.find("response_format"), std::string::npos);
  EXPECT_NE(req.body.find("name=\"stream\""), std::string::npos);
  EXPECT_NE(req.body.find("\r\n\r\nfalse"), std::string::npos);
  EXPECT_NE(req.body.find("temperature"), std::string::npos);
  EXPECT_NE(req.body.find("name=\"timestamp_granularities[]\""), std::string::npos);

  std::filesystem::remove(tmp);
}

TEST(AudioTranslationsResourceTest, CreateParsesResponse) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  const std::string response_body =
      R"({"text":"translated text","duration":2.0,"language":"english","segments":[{"id":0,"avg_logprob":-0.2,"compression_ratio":0.6,"end":1.5,"no_speech_prob":0.02,"seek":0,"start":0.0,"temperature":0.0,"text":"translated text","tokens":[10,11]}]})";
  mock_ptr->enqueue_response(HttpResponse{200, {}, response_body});

  std::filesystem::path tmp = std::filesystem::temp_directory_path() / "openai-cpp-audio-translate.wav";
  {
    std::ofstream out(tmp, std::ios::binary);
    out << "audio";
  }

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  TranslationRequest request;
  request.file.purpose = "assistants";
  request.file.file_path = tmp.string();
  request.file.file_name = "audio.wav";
  request.file.content_type = "audio/wav";
  request.model = "whisper-1";
  request.response_format = "verbose_json";

  auto translation = client.audio().translations().create(request);
  EXPECT_EQ(translation.text, "translated text");
  ASSERT_TRUE(translation.duration.has_value());
  EXPECT_DOUBLE_EQ(translation.duration.value(), 2.0);
  ASSERT_TRUE(translation.language.has_value());
  EXPECT_EQ(*translation.language, "english");
  ASSERT_TRUE(translation.segments.has_value());
  EXPECT_EQ(translation.segments->at(0).text, "translated text");
  EXPECT_TRUE(translation.is_verbose);
  EXPECT_FALSE(translation.is_plain_text);

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  const auto& req = *mock_ptr->last_request();
  EXPECT_NE(req.headers.at("Content-Type").find("multipart/form-data"), std::string::npos);

  std::filesystem::remove(tmp);
}

TEST(AudioSpeechResourceTest, CreateReturnsBinaryAudio) {
  using namespace openai;

  auto mock_client = std::make_unique<oait::MockHttpClient>();
  auto* mock_ptr = mock_client.get();

  mock_ptr->enqueue_response(HttpResponse{200, {{"Content-Type", "application/octet-stream"}}, std::string("AUDIO")});

  ClientOptions options;
  options.api_key = "sk-test";

  OpenAIClient client(options, std::move(mock_client));

  SpeechRequest request;
  request.input = "Hello world";
  request.model = "tts-1";
  request.voice = "alloy";

  auto speech = client.audio().speech().create(request);
  ASSERT_EQ(speech.audio.size(), 5u);
  EXPECT_EQ(std::string(speech.audio.begin(), speech.audio.end()), "AUDIO");
  ASSERT_TRUE(speech.headers.count("Content-Type"));
  EXPECT_EQ(speech.headers.at("Content-Type"), "application/octet-stream");

  ASSERT_TRUE(mock_ptr->last_request().has_value());
  EXPECT_EQ(mock_ptr->last_request()->headers.at("Accept"), "application/octet-stream");
}
