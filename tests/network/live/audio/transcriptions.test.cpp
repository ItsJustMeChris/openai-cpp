#include <gtest/gtest.h>

#include "openai/audio.hpp"
#include "openai/client.hpp"
#include "openai/error.hpp"
#include "network/live/audio/audio_live_test_utils.hpp"
#include "network/live/live_test_utils.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

using openai::test::live::env_flag_enabled;
using openai::test::live::get_env;
using openai::test::live::make_live_client_options;
using openai::test::live::audio::make_audio_upload;
using openai::test::live::audio::synthesize_speech_file;
using openai::test::live::audio::TempBinaryFile;

namespace
{

  std::string transcription_model()
  {
    return get_env("OPENAI_CPP_LIVE_TRANSCRIPTION_MODEL").value_or("gpt-4o-transcribe");
  }

} // namespace

TEST(AudioTranscriptionsLiveNetworkTest, CreateOnlyRequiredParams)
{
  if (!env_flag_enabled("OPENAI_CPP_ENABLE_LIVE_TESTS"))
  {
    GTEST_SKIP() << "Set OPENAI_CPP_ENABLE_LIVE_TESTS=1 to enable live OpenAI API tests.";
  }

  auto options = make_live_client_options();
  if (!options)
  {
    GTEST_SKIP() << "OPENAI_API_KEY is not set; skipping live OpenAI API tests.";
  }
  openai::OpenAIClient client(std::move(*options));

  openai::TranscriptionRequest request;
  TempBinaryFile synthesized = [&]() -> TempBinaryFile
  {
    try
    {
      return synthesize_speech_file(client,
                                    "transcription-basic",
                                    "Hello from the OpenAI C plus plus live transcription test.",
                                    "wav",
                                    1.0);
    }
    catch (const openai::APIError &err)
    {
      ADD_FAILURE() << "audio.speech.create (for transcription fixture) failed (status " << err.status_code()
                    << "): " << err.what();
      throw;
    }
    catch (const openai::OpenAIError &err)
    {
      ADD_FAILURE() << "Unexpected error during audio.speech.create (for transcription fixture): " << err.what();
      throw;
    }
  }();

  request.file = make_audio_upload(synthesized.path(), "audio/wav");
  request.model = transcription_model();

  try
  {
    const auto transcription = client.audio().transcriptions().create(request);
    EXPECT_FALSE(transcription.text.empty());
    if (transcription.usage)
    {
      EXPECT_GT(transcription.usage->total_tokens, 0);
    }
  }
  catch (const openai::APIError &err)
  {
    FAIL() << "audio.transcriptions.create failed (status " << err.status_code() << "): " << err.what();
  }
  catch (const openai::OpenAIError &err)
  {
    FAIL() << "Unexpected error during audio.transcriptions.create: " << err.what();
  }
}

TEST(AudioTranscriptionsLiveNetworkTest, CreateWithOptionalParams)
{
  if (!env_flag_enabled("OPENAI_CPP_ENABLE_LIVE_TESTS"))
  {
    GTEST_SKIP() << "Set OPENAI_CPP_ENABLE_LIVE_TESTS=1 to enable live OpenAI API tests.";
  }

  auto options = make_live_client_options();
  if (!options)
  {
    GTEST_SKIP() << "OPENAI_API_KEY is not set; skipping live OpenAI API tests.";
  }
  openai::OpenAIClient client(std::move(*options));

  const std::string synthesis_text = "Please transcribe this audio for the OpenAI C plus plus live test.";
  TempBinaryFile synthesized = [&]() -> TempBinaryFile
  {
    try
    {
      return synthesize_speech_file(client, "transcription-extended", synthesis_text, "wav", 1.0);
    }
    catch (const openai::APIError &err)
    {
      ADD_FAILURE() << "audio.speech.create (for transcription fixture) failed (status " << err.status_code()
                    << "): " << err.what();
      throw;
    }
    catch (const openai::OpenAIError &err)
    {
      ADD_FAILURE() << "Unexpected error during audio.speech.create (for transcription fixture): " << err.what();
      throw;
    }
  }();

  openai::TranscriptionRequest request;
  request.file = make_audio_upload(synthesized.path(), "audio/wav");
  request.model = transcription_model();
  openai::TranscriptionChunkingStrategy strategy;
  strategy.type = openai::TranscriptionChunkingStrategy::Type::Auto;
  request.chunking_strategy = strategy;
  request.include = std::vector<std::string>{"logprobs"};
  request.language = std::string("en");
  request.prompt = std::string("Transcribe the spoken content precisely.");
  request.response_format = std::string("json");
  request.stream = false;
  request.temperature = 0.0;
  request.timestamp_granularities = std::vector<std::string>{"word"};

  try
  {
    const auto transcription = client.audio().transcriptions().create(request);
    EXPECT_FALSE(transcription.text.empty());
    std::string normalized = transcription.text;
    std::transform(normalized.begin(),
                   normalized.end(),
                   normalized.begin(),
                   [](unsigned char ch)
                   { return static_cast<char>(std::tolower(ch)); });
    EXPECT_NE(normalized.find("openai"), std::string::npos);
    if (transcription.usage)
    {
      EXPECT_GE(transcription.usage->input_tokens, 0);
    }
    EXPECT_TRUE(transcription.segments.has_value() || transcription.diarized_segments.has_value() || transcription.text.length() > 0);
  }
  catch (const openai::APIError &err)
  {
    FAIL() << "audio.transcriptions.create (with optional params) failed (status " << err.status_code()
           << "): " << err.what();
  }
  catch (const openai::OpenAIError &err)
  {
    FAIL() << "Unexpected error during audio.transcriptions.create (with optional params): " << err.what();
  }
}
