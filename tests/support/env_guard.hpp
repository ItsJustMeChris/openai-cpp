#pragma once

#include <cstdlib>
#include <optional>
#include <string>

namespace openai::testing {

inline void set_env(const std::string& name, const std::string& value) {
#if defined(_WIN32)
  _putenv_s(name.c_str(), value.c_str());
#else
  ::setenv(name.c_str(), value.c_str(), 1);
#endif
}

inline void unset_env(const std::string& name) {
#if defined(_WIN32)
  _putenv_s(name.c_str(), "");
#else
  ::unsetenv(name.c_str());
#endif
}

class EnvVarGuard {
public:
  EnvVarGuard(std::string name, std::optional<std::string> value)
      : name_(std::move(name)) {
    const char* existing = std::getenv(name_.c_str());
    if (existing != nullptr) {
      previous_ = std::string(existing);
    }
    if (value.has_value()) {
      set_env(name_, *value);
    } else {
      unset_env(name_);
    }
  }

  EnvVarGuard(const EnvVarGuard&) = delete;
  EnvVarGuard& operator=(const EnvVarGuard&) = delete;

  EnvVarGuard(EnvVarGuard&& other) noexcept
      : name_(std::move(other.name_)), previous_(std::move(other.previous_)), active_(other.active_) {
    other.active_ = false;
  }

  EnvVarGuard& operator=(EnvVarGuard&& other) noexcept {
    if (this != &other) {
      reset();
      name_ = std::move(other.name_);
      previous_ = std::move(other.previous_);
      active_ = other.active_;
      other.active_ = false;
    }
    return *this;
  }

  ~EnvVarGuard() { reset(); }

private:
  void reset() {
    if (!active_) {
      return;
    }
    if (previous_.has_value()) {
      set_env(name_, *previous_);
    } else {
      unset_env(name_);
    }
    active_ = false;
  }

  std::string name_;
  std::optional<std::string> previous_;
  bool active_ = true;
};

}  // namespace openai::testing

