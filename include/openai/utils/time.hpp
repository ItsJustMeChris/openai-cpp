#pragma once

#include <chrono>
#include <optional>

namespace openai::utils {

void sleep_for(std::chrono::milliseconds duration);

double retry_jitter_factor();

std::chrono::milliseconds calculate_default_retry_delay(std::size_t retries_remaining,
                                                        std::size_t max_retries,
                                                        std::optional<double> jitter_factor = std::nullopt);

}  // namespace openai::utils
