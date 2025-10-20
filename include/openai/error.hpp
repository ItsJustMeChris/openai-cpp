#pragma once

#include <stdexcept>
#include <string>

namespace openai {

class OpenAIError : public std::runtime_error {
public:
  explicit OpenAIError(const std::string& message)
      : std::runtime_error(message) {}
};

class HttpError : public OpenAIError {
public:
  HttpError(long status_code, const std::string& message)
      : OpenAIError(message), status_code_(status_code) {}

  long status_code() const { return status_code_; }

private:
  long status_code_;
};

}  // namespace openai

