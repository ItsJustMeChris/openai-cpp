#include "openai/utils/time.hpp"

#include <cmath>
#include <random>
#include <thread>

namespace openai::utils {
namespace {

constexpr double kInitialRetryDelaySeconds = 0.5;
constexpr double kMaxRetryDelaySeconds = 8.0;

}  // namespace

void sleep_for(std::chrono::milliseconds duration) {
  if (duration.count() <= 0) {
    return;
  }
  std::this_thread::sleep_for(duration);
}

double retry_jitter_factor() {
  thread_local std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<double> dist(0.0, 0.25);
  return 1.0 - dist(rng);
}

std::chrono::milliseconds calculate_default_retry_delay(std::size_t retries_remaining,
                                                        std::size_t max_retries,
                                                        std::optional<double> jitter_factor) {
  double jitter = jitter_factor.value_or(retry_jitter_factor());
  if (jitter < 0.0) {
    jitter = 0.0;
  }

  double sleep_seconds;
  if (max_retries == 0) {
    sleep_seconds = kInitialRetryDelaySeconds;
    jitter = 1.0;
  } else {
    std::size_t num_retries = max_retries > retries_remaining ? max_retries - retries_remaining : 0;
    sleep_seconds = kInitialRetryDelaySeconds * std::pow(2.0, static_cast<double>(num_retries));
    if (sleep_seconds > kMaxRetryDelaySeconds) {
      sleep_seconds = kMaxRetryDelaySeconds;
    }
  }

  sleep_seconds *= jitter;
  if (sleep_seconds < 0.0) {
    sleep_seconds = 0.0;
  }

  return std::chrono::milliseconds(static_cast<long>(sleep_seconds * 1000.0));
}

}  // namespace openai::utils
