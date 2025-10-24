#include <gtest/gtest.h>

#include "openai/audio.hpp"
#include "openai/client.hpp"
#include "openai/error.hpp"
#include "network/live/live_test_utils.hpp"

#include <string>

using openai::test::live::env_flag_enabled;
using openai::test::live::get_env;
using openai::test::live::make_live_client_options;

namespace {

std::string speech_model() {
  return get_env("OPENAI_CPP_LIVE_SPEECH_MODEL").value_or("gpt-4o-mini-tts");
}

std::string speech_voice() {
  return get_env("OPENAI_CPP_LIVE_SPEECH_VOICE").value_or("alloy");
}

}  // namespace

TEST(AudioSpeechLiveNetworkTest, CreateReturnsAudio) {
  if (!env_flag_enabled("OPENAI_CPP_ENABLE_LIVE_TESTS")) {
    GTEST_SKIP() << "Set OPENAI_CPP_ENABLE_LIVE_TESTS=1 to enable live OpenAI API tests.";
  }

  auto options = make_live_client_options();
  if (!options) {
    GTEST_SKIP() << "OPENAI_API_KEY is not set; skipping live OpenAI API tests.";
  }
  openai::OpenAIClient client(std::move(*options));

  openai::SpeechRequest request;
  request.input = "Hello from the OpenAI C++ live audio test suite.";
  request.model = speech_model();
  request.voice = speech_voice();
  request.instructions = std::string("Deliver the line with a calm tone.");
  request.response_format = std::string("mp3");
  request.speed = 1.0;

  try {
    const auto speech = client.audio().speech().create(request);
    EXPECT_FALSE(speech.audio.empty());
    EXPECT_GE(speech.audio.size(), 128u);
    EXPECT_FALSE(speech.headers.empty());
    if (speech.headers.count("Content-Type")) {
      EXPECT_FALSE(speech.headers.at("Content-Type").empty());
    }
  } catch (const openai::APIError& err) {
    FAIL() << "audio.speech.create failed (status " << err.status_code() << "): " << err.what();
  } catch (const openai::OpenAIError& err) {
    FAIL() << "Unexpected error during audio.speech.create: " << err.what();
  }
}

