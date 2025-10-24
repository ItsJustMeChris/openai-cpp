#include <gtest/gtest.h>

#include "openai/audio.hpp"
#include "openai/client.hpp"
#include "openai/error.hpp"
#include "network/live/audio/audio_live_test_utils.hpp"
#include "network/live/live_test_utils.hpp"

#include <string>

using openai::test::live::audio::TempBinaryFile;
using openai::test::live::audio::make_audio_upload;
using openai::test::live::audio::synthesize_speech_file;
using openai::test::live::env_flag_enabled;
using openai::test::live::get_env;
using openai::test::live::make_live_client_options;

namespace {

std::string translation_model() {
  return get_env("OPENAI_CPP_LIVE_TRANSLATION_MODEL").value_or("whisper-1");
}

}  // namespace

TEST(AudioTranslationsLiveNetworkTest, CreateOnlyRequiredParams) {
  if (!env_flag_enabled("OPENAI_CPP_ENABLE_LIVE_TESTS")) {
    GTEST_SKIP() << "Set OPENAI_CPP_ENABLE_LIVE_TESTS=1 to enable live OpenAI API tests.";
  }

  auto options = make_live_client_options();
  if (!options) {
    GTEST_SKIP() << "OPENAI_API_KEY is not set; skipping live OpenAI API tests.";
  }
  openai::OpenAIClient client(std::move(*options));

  TempBinaryFile synthesized = [&]() -> TempBinaryFile {
    try {
      return synthesize_speech_file(client,
                                    "translation-basic",
                                    "Please translate this audio for the OpenAI C plus plus live test.",
                                    "wav",
                                    1.0);
    } catch (const openai::APIError& err) {
      ADD_FAILURE() << "audio.speech.create (for translation fixture) failed (status " << err.status_code()
                    << "): " << err.what();
      throw;
    } catch (const openai::OpenAIError& err) {
      ADD_FAILURE() << "Unexpected error during audio.speech.create (for translation fixture): " << err.what();
      throw;
    }
  }();

  openai::TranslationRequest request;
  request.file = make_audio_upload(synthesized.path(), "audio/wav");
  request.model = translation_model();

  try {
    const auto translation = client.audio().translations().create(request);
    EXPECT_FALSE(translation.text.empty());
  } catch (const openai::APIError& err) {
    FAIL() << "audio.translations.create failed (status " << err.status_code() << "): " << err.what();
  } catch (const openai::OpenAIError& err) {
    FAIL() << "Unexpected error during audio.translations.create: " << err.what();
  }
}

TEST(AudioTranslationsLiveNetworkTest, CreateWithOptionalParams) {
  if (!env_flag_enabled("OPENAI_CPP_ENABLE_LIVE_TESTS")) {
    GTEST_SKIP() << "Set OPENAI_CPP_ENABLE_LIVE_TESTS=1 to enable live OpenAI API tests.";
  }

  auto options = make_live_client_options();
  if (!options) {
    GTEST_SKIP() << "OPENAI_API_KEY is not set; skipping live OpenAI API tests.";
  }
  openai::OpenAIClient client(std::move(*options));

  TempBinaryFile synthesized = [&]() -> TempBinaryFile {
    try {
      return synthesize_speech_file(client,
                                    "translation-extended",
                                    "Please translate this audio for the OpenAI C plus plus live test.",
                                    "wav",
                                    1.0);
    } catch (const openai::APIError& err) {
      ADD_FAILURE() << "audio.speech.create (for translation fixture) failed (status " << err.status_code()
                    << "): " << err.what();
      throw;
    } catch (const openai::OpenAIError& err) {
      ADD_FAILURE() << "Unexpected error during audio.speech.create (for translation fixture): " << err.what();
      throw;
    }
  }();

  openai::TranslationRequest request;
  request.file = make_audio_upload(synthesized.path(), "audio/wav");
  request.model = translation_model();
  request.prompt = std::string("Translate the provided speech into English.");
  request.response_format = std::string("json");
  request.temperature = 0.0;

  try {
    const auto translation = client.audio().translations().create(request);
    EXPECT_FALSE(translation.text.empty());
    if (translation.language) {
      EXPECT_FALSE(translation.language->empty());
    }
  } catch (const openai::APIError& err) {
    FAIL() << "audio.translations.create (with optional params) failed (status " << err.status_code()
           << "): " << err.what();
  } catch (const openai::OpenAIError& err) {
    FAIL() << "Unexpected error during audio.translations.create (with optional params): " << err.what();
  }
}
