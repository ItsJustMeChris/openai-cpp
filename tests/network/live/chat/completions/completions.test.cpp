#include <gtest/gtest.h>

#include "openai/chat.hpp"
#include "openai/client.hpp"
#include "openai/error.hpp"
#include "openai/models.hpp"
#include "network/live/chat/completions/live_test_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using openai::test::live::env_flag_enabled;
using openai::test::live::get_env;
using openai::test::live::make_live_client_options;
using openai::test::live::make_text_message;
using openai::test::live::unique_tag;

TEST(ChatCompletionsLiveNetworkTest, ListModelsReturnsResults) {
  if (!env_flag_enabled("OPENAI_CPP_ENABLE_LIVE_TESTS")) {
    GTEST_SKIP() << "Set OPENAI_CPP_ENABLE_LIVE_TESTS=1 to enable live OpenAI API tests.";
  }

  auto options = make_live_client_options();
  if (!options) {
    GTEST_SKIP() << "OPENAI_API_KEY is not set; skipping live OpenAI API tests.";
  }
  openai::OpenAIClient client(std::move(*options));

  try {
    const auto models = client.models().list();
    ASSERT_FALSE(models.data.empty());
    EXPECT_FALSE(models.data.front().id.empty());
  } catch (const openai::APIError& err) {
    FAIL() << "Live models.list failed (status " << err.status_code() << "): " << err.what();
  } catch (const openai::OpenAIError& err) {
    FAIL() << "Unexpected error during models.list: " << err.what();
  }
}

TEST(ChatCompletionsLiveNetworkTest, CreateOnlyRequiredParams) {
  if (!env_flag_enabled("OPENAI_CPP_ENABLE_LIVE_TESTS")) {
    GTEST_SKIP() << "Set OPENAI_CPP_ENABLE_LIVE_TESTS=1 to enable live OpenAI API tests.";
  }

  auto options = make_live_client_options();
  if (!options) {
    GTEST_SKIP() << "OPENAI_API_KEY is not set; skipping live OpenAI API tests.";
  }
  openai::OpenAIClient client(std::move(*options));

  openai::ChatCompletionRequest request;
  request.model = get_env("OPENAI_CPP_LIVE_TEST_MODEL").value_or("gpt-4o");
  request.messages.push_back(make_text_message("system", "You are a concise assistant for integration testing."));
  request.messages.push_back(make_text_message("user", "Respond with a single word greeting."));
  request.max_tokens = 16;
  request.temperature = 0.2;

  try {
    const auto completion = client.chat().completions().create(request);
    EXPECT_FALSE(completion.id.empty());
    ASSERT_FALSE(completion.choices.empty());
    const auto& choice = completion.choices.front();
    ASSERT_TRUE(choice.message.has_value());
    const auto& message = *choice.message;
    ASSERT_FALSE(message.content.empty());
    const auto& first_block = message.content.front();
    EXPECT_EQ(first_block.type, openai::ChatMessageContent::Type::Text);
    EXPECT_FALSE(first_block.text.empty());
    if (completion.usage) {
      EXPECT_GT(completion.usage->total_tokens, 0);
    }
  } catch (const openai::APIError& err) {
    FAIL() << "Live chat.completions.create failed (status " << err.status_code() << "): " << err.what();
  } catch (const openai::OpenAIError& err) {
    FAIL() << "Unexpected error during chat.completions.create: " << err.what();
  }
}

TEST(ChatCompletionsLiveNetworkTest, CreateWithExtendedParams) {
  if (!env_flag_enabled("OPENAI_CPP_ENABLE_LIVE_TESTS")) {
    GTEST_SKIP() << "Set OPENAI_CPP_ENABLE_LIVE_TESTS=1 to enable live OpenAI API tests.";
  }

  auto options = make_live_client_options();
  if (!options) {
    GTEST_SKIP() << "OPENAI_API_KEY is not set; skipping live OpenAI API tests.";
  }
  openai::OpenAIClient client(std::move(*options));

  const std::string tag = unique_tag();

  openai::ChatCompletionRequest request;
  request.model = get_env("OPENAI_CPP_LIVE_TEST_MODEL").value_or("gpt-4o");
  request.messages.push_back(make_text_message("system", "You are a compliance tester. Reply briefly."));
  request.messages.push_back(make_text_message("user", "State the live test tag exactly: " + tag));
  request.max_tokens = 32;
  request.temperature = 0.1;
  request.top_p = 0.9;
  request.frequency_penalty = 0.0;
  request.presence_penalty = 0.0;
  request.stop = std::vector<std::string>{"<END>"};
  request.seed = 42;
  request.store = true;
  request.user = "openai-cpp-live-test";
  request.metadata = {{"test-suite", "resource-completions"}, {"test-tag", tag}};
  request.modalities = {"text"};

  try {
    const auto completion = client.chat().completions().create(request);
    EXPECT_FALSE(completion.id.empty());
    ASSERT_FALSE(completion.choices.empty());
    const auto& choice = completion.choices.front();
    ASSERT_TRUE(choice.message.has_value());
    const auto& message = *choice.message;
    ASSERT_FALSE(message.content.empty());
    const auto& first_block = message.content.front();
    EXPECT_EQ(first_block.type, openai::ChatMessageContent::Type::Text);
    EXPECT_FALSE(first_block.text.empty());
    if (!completion.metadata.empty()) {
      auto it = completion.metadata.find("test-tag");
      if (it != completion.metadata.end()) {
        EXPECT_EQ(it->second, tag);
      }
    }
    if (completion.usage) {
      EXPECT_GT(completion.usage->prompt_tokens, 0);
      EXPECT_GT(completion.usage->completion_tokens, 0);
    }
  } catch (const openai::APIError& err) {
    FAIL() << "Live chat.completions.create with extended params failed (status " << err.status_code()
           << "): " << err.what();
  } catch (const openai::OpenAIError& err) {
    FAIL() << "Unexpected error during extended chat.completions.create: " << err.what();
  }
}

TEST(ChatCompletionsLiveNetworkTest, RetrieveUpdateListAndDeleteStoredCompletion) {
  if (!env_flag_enabled("OPENAI_CPP_ENABLE_LIVE_TESTS")) {
    GTEST_SKIP() << "Set OPENAI_CPP_ENABLE_LIVE_TESTS=1 to enable live OpenAI API tests.";
  }

  auto options = make_live_client_options();
  if (!options) {
    GTEST_SKIP() << "OPENAI_API_KEY is not set; skipping live OpenAI API tests.";
  }
  openai::OpenAIClient client(std::move(*options));

  const std::string initial_tag = unique_tag();
  const std::string updated_tag = unique_tag();

  openai::ChatCompletionRequest request;
  request.model = get_env("OPENAI_CPP_LIVE_TEST_MODEL").value_or("gpt-4o");
  request.messages.push_back(make_text_message("system", "You are a stateful integration tester."));
  request.messages.push_back(make_text_message("user", "Acknowledge with the tag: " + initial_tag));
  request.store = true;
  request.max_tokens = 32;
  request.metadata = {{"test-suite", "resource-completions"}, {"stage", "initial"}, {"test-tag", initial_tag}};

  openai::ChatCompletion completion;

  try {
    completion = client.chat().completions().create(request);
  } catch (const openai::APIError& err) {
    FAIL() << "Failed to create stored completion (status " << err.status_code() << "): " << err.what();
  } catch (const openai::OpenAIError& err) {
    FAIL() << "Unexpected error while creating stored completion: " << err.what();
  }

  ASSERT_FALSE(completion.id.empty());
  const std::string completion_id = completion.id;

  SCOPED_TRACE("Stored completion id: " + completion_id);

  bool needs_cleanup = true;
  auto cleanup = [&]() {
    if (!needs_cleanup) {
      return;
    }
    try {
      client.chat().completions().remove(completion_id);
      needs_cleanup = false;
    } catch (...) {
      // Ignore cleanup errors.
    }
  };

  sleep(5);

  openai::ChatCompletion retrieved;
  try {
    retrieved = client.chat().completions().retrieve(completion_id);
  } catch (const openai::APIError& err) {
    cleanup();
    FAIL() << "chat.completions.retrieve(" << completion_id << ") failed (status " << err.status_code()
           << "): " << err.what();
  } catch (const openai::OpenAIError& err) {
    cleanup();
    FAIL() << "Unexpected error during chat.completions.retrieve(" << completion_id << "): " << err.what();
  }
  EXPECT_EQ(retrieved.id, completion_id);

  openai::ChatCompletionUpdateRequest update_request;
  update_request.metadata = std::map<std::string, std::string>{
      {"test-suite", "resource-completions"},
      {"stage", "updated"},
      {"test-tag", updated_tag},
  };

  openai::ChatCompletion updated;
  try {
    updated = client.chat().completions().update(completion_id, update_request);
  } catch (const openai::APIError& err) {
    cleanup();
    FAIL() << "chat.completions.update(" << completion_id << ") failed (status " << err.status_code()
           << "): " << err.what();
  } catch (const openai::OpenAIError& err) {
    cleanup();
    FAIL() << "Unexpected error during chat.completions.update(" << completion_id << "): " << err.what();
  }
  if (!updated.metadata.empty()) {
    auto it = updated.metadata.find("stage");
    if (it != updated.metadata.end()) {
      EXPECT_EQ(it->second, "updated");
    }
  }

  openai::ChatCompletionListParams list_params;
  list_params.limit = 20;
  list_params.metadata = std::map<std::string, std::string>{{"test-tag", updated_tag}};

  openai::ChatCompletionList list;
  try {
    list = client.chat().completions().list(list_params);
    /*
    this fails because openais api is broken. even the docs show get and running the curl demo fails. 
curl https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json"

    */
  } catch (const openai::APIError& err) {
    cleanup();
    FAIL() << "chat.completions.list failed while filtering for test-tag=" << updated_tag << " (status "
           << err.status_code() << "): " << err.what();
  } catch (const openai::OpenAIError& err) {
    cleanup();
    FAIL() << "Unexpected error during chat.completions.list while filtering for test-tag=" << updated_tag
           << ": " << err.what();
  }
  bool found = false;
  for (const auto& item : list.data) {
    if (item.id == completion_id) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);

  openai::ChatCompletionDeleted deleted;
  try {
    deleted = client.chat().completions().remove(completion_id);
    needs_cleanup = false;
  } catch (const openai::APIError& err) {
    cleanup();
    FAIL() << "chat.completions.remove(" << completion_id << ") failed (status " << err.status_code()
           << "): " << err.what();
  } catch (const openai::OpenAIError& err) {
    cleanup();
    FAIL() << "Unexpected error during chat.completions.remove(" << completion_id << "): " << err.what();
  }
  EXPECT_EQ(deleted.id, completion_id);
  EXPECT_TRUE(deleted.deleted);
}
