#pragma once

#include <map>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace openai {

class OpenAIError : public std::runtime_error {
public:
  explicit OpenAIError(const std::string& message)
      : std::runtime_error(message) {}
};

class APIError : public OpenAIError {
public:
  APIError(std::string message,
           long status_code,
           nlohmann::json error_body,
           std::map<std::string, std::string> headers)
      : OpenAIError(std::move(message)),
        status_code_(status_code),
        error_body_(std::move(error_body)),
        headers_(std::move(headers)) {}

  long status_code() const { return status_code_; }
  const nlohmann::json& error_body() const { return error_body_; }
  const std::map<std::string, std::string>& headers() const { return headers_; }

private:
  long status_code_;
  nlohmann::json error_body_;
  std::map<std::string, std::string> headers_;
};

class BadRequestError : public APIError {
public:
  using APIError::APIError;
};

class AuthenticationError : public APIError {
public:
  using APIError::APIError;
};

class PermissionDeniedError : public APIError {
public:
  using APIError::APIError;
};

class NotFoundError : public APIError {
public:
  using APIError::APIError;
};

class ConflictError : public APIError {
public:
  using APIError::APIError;
};

class UnprocessableEntityError : public APIError {
public:
  using APIError::APIError;
};

class RateLimitError : public APIError {
public:
  using APIError::APIError;
};

class InternalServerError : public APIError {
public:
  using APIError::APIError;
};

class APIConnectionError : public OpenAIError {
public:
  explicit APIConnectionError(const std::string& message)
      : OpenAIError(message) {}
};

class APIConnectionTimeoutError : public APIConnectionError {
public:
  using APIConnectionError::APIConnectionError;
};

class APIUserAbortError : public OpenAIError {
public:
  explicit APIUserAbortError(const std::string& message)
      : OpenAIError(message) {}
};

class HttpError : public APIError {
public:
  using APIError::APIError;
};

class OpenAIInvalidHeaderError : public OpenAIError {
public:
  explicit OpenAIInvalidHeaderError(const std::string& message)
      : OpenAIError(message) {}
};

}  // namespace openai
