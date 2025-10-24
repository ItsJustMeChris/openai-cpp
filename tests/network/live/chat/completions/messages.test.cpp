#include <gtest/gtest.h>

#include "openai/chat.hpp"
#include "openai/client.hpp"
#include "openai/error.hpp"
#include "network/live/chat/completions/live_test_utils.hpp"

#include <chrono>
#include <map>
#include <optional>
#include <string>
#include <thread>

using openai::test::live::env_flag_enabled;
using openai::test::live::get_env;
using openai::test::live::make_live_client_options;
using openai::test::live::make_text_message;
using openai::test::live::unique_tag;

namespace {

class CompletionCleanup {
public:
  CompletionCleanup(openai::OpenAIClient& client, std::string completion_id)
      : client_(client), completion_id_(std::move(completion_id)) {}

  CompletionCleanup(const CompletionCleanup&) = delete;
  CompletionCleanup& operator=(const CompletionCleanup&) = delete;
  CompletionCleanup(CompletionCleanup&&) = delete;
  CompletionCleanup& operator=(CompletionCleanup&&) = delete;

  ~CompletionCleanup() {
    if (!completion_id_.empty()) {
      try {
        client_.chat().completions().remove(completion_id_);
      } catch (...) {
        // Best-effort cleanup; ignore failures.
      }
    }
  }

private:
  openai::OpenAIClient& client_;
  std::string completion_id_;
};

struct StoredCompletionResult {
  openai::ChatCompletion completion;
  std::string tag;
  std::string prompt;
};

StoredCompletionResult create_stored_completion(openai::OpenAIClient& client,
                                                const std::string& purpose,
                                                std::optional<std::string> extra_metadata = std::nullopt) {
  openai::ChatCompletionRequest request;
  request.model = get_env("OPENAI_CPP_LIVE_TEST_MODEL").value_or("gpt-4o");
  const std::string tag = unique_tag();
  const std::string prompt = "Respond succinctly with the tag: " + tag;
  request.messages.push_back(make_text_message("system", "You are assisting with integration tests."));
  request.messages.push_back(make_text_message("user", prompt));
  request.store = true;
  request.max_tokens = 32;

  std::map<std::string, std::string> metadata = {
      {"test-suite", "resource-messages"},
      {"test-purpose", purpose},
      {"tag", tag},
  };
  if (extra_metadata) {
    metadata.emplace("extra", *extra_metadata);
  }
  request.metadata = std::move(metadata);

  StoredCompletionResult result;
  result.completion = client.chat().completions().create(request);
  result.tag = tag;
  result.prompt = prompt;
  return result;
}

bool contains_user_prompt(const openai::ChatCompletionStoreMessageList& messages,
                          const std::string& expected_prompt) {
  for (const auto& entry : messages.data) {
    if (entry.message.role != "user") {
      continue;
    }
    for (const auto& part : entry.message.content) {
      if (part.type == openai::ChatMessageContent::Type::Text && part.text == expected_prompt) {
        return true;
      }
    }
    for (const auto& part : entry.content_parts) {
      if (part.type == openai::ChatMessageContent::Type::Text && part.text == expected_prompt) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace

TEST(ChatCompletionsMessagesLiveNetworkTest, ListReturnsStoredMessages) {
  if (!env_flag_enabled("OPENAI_CPP_ENABLE_LIVE_TESTS")) {
    GTEST_SKIP() << "Set OPENAI_CPP_ENABLE_LIVE_TESTS=1 to enable live OpenAI API tests.";
  }

  auto options = make_live_client_options();
  if (!options) {
    GTEST_SKIP() << "OPENAI_API_KEY is not set; skipping live OpenAI API tests.";
  }
  openai::OpenAIClient client(std::move(*options));

  StoredCompletionResult stored;
  try {
    stored = create_stored_completion(client, "list");
  } catch (const openai::APIError& err) {
    FAIL() << "Failed to create stored completion (status " << err.status_code() << "): " << err.what();
  } catch (const openai::OpenAIError& err) {
    FAIL() << "Unexpected error while creating stored completion: " << err.what();
  }

  const auto& completion = stored.completion;

  ASSERT_FALSE(completion.id.empty());
  const std::string completion_id = completion.id;
  CompletionCleanup cleanup(client, completion_id);

  std::this_thread::sleep_for(std::chrono::seconds(5));

  openai::ChatCompletionStoreMessageList messages;
  try {
    messages = client.chat().completions().messages().list(completion_id);
  } catch (const openai::APIError& err) {
    FAIL() << "chat.completions.messages.list(" << completion_id << ") failed (status " << err.status_code()
           << "): " << err.what();
  } catch (const openai::OpenAIError& err) {
    FAIL() << "Unexpected error during chat.completions.messages.list(" << completion_id << "): " << err.what();
  }

  EXPECT_FALSE(messages.data.empty());
  EXPECT_TRUE(contains_user_prompt(messages, stored.prompt));
}

TEST(ChatCompletionsMessagesLiveNetworkTest, ListHonorsParamsAndRequestOptions) {
  if (!env_flag_enabled("OPENAI_CPP_ENABLE_LIVE_TESTS")) {
    GTEST_SKIP() << "Set OPENAI_CPP_ENABLE_LIVE_TESTS=1 to enable live OpenAI API tests.";
  }

  auto options = make_live_client_options();
  if (!options) {
    GTEST_SKIP() << "OPENAI_API_KEY is not set; skipping live OpenAI API tests.";
  }
  openai::OpenAIClient client(std::move(*options));

  StoredCompletionResult stored;
  try {
    stored = create_stored_completion(client, "options", std::optional<std::string>("params"));
  } catch (const openai::APIError& err) {
    FAIL() << "Failed to create stored completion (status " << err.status_code() << "): " << err.what();
  } catch (const openai::OpenAIError& err) {
    FAIL() << "Unexpected error while creating stored completion: " << err.what();
  }

  const auto& completion = stored.completion;

  ASSERT_FALSE(completion.id.empty());
  CompletionCleanup cleanup(client, completion.id);

  std::this_thread::sleep_for(std::chrono::seconds(5));

  openai::ChatCompletionMessageListParams params;
  params.limit = 1;
  params.order = std::string("asc");

  openai::RequestOptions request_options;
  request_options.headers["Authorization"] = std::string("Bearer invalid-live-test-token");
  request_options.query_params["extra"] = std::string("value");

  EXPECT_THROW(
      client.chat().completions().messages().list(completion.id, params, request_options),
      openai::AuthenticationError);
}
