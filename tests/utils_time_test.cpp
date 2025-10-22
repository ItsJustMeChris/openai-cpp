#include <gtest/gtest.h>

#include <chrono>

#include "openai/utils/time.hpp"

using openai::utils::calculate_default_retry_delay;
using openai::utils::retry_jitter_factor;
using openai::utils::sleep_for;

TEST(UtilsTimeTest, RetryDelayUsesExponentialBackoff) {
  auto first = calculate_default_retry_delay(/*retries_remaining=*/3, /*max_retries=*/3, 1.0);
  auto second = calculate_default_retry_delay(/*retries_remaining=*/2, /*max_retries=*/3, 1.0);
  auto third = calculate_default_retry_delay(/*retries_remaining=*/1, /*max_retries=*/3, 1.0);

  EXPECT_EQ(first, std::chrono::milliseconds(500));
  EXPECT_EQ(second, std::chrono::milliseconds(1000));
  EXPECT_EQ(third, std::chrono::milliseconds(2000));
}

TEST(UtilsTimeTest, RetryDelayClampsToMaximum) {
  auto delay = calculate_default_retry_delay(/*retries_remaining=*/0, /*max_retries=*/10, 1.0);
  EXPECT_EQ(delay, std::chrono::milliseconds(8000));
}

TEST(UtilsTimeTest, RetryDelayWithZeroMaxRetriesStillUsesInitialDelay) {
  auto delay = calculate_default_retry_delay(/*retries_remaining=*/0, /*max_retries=*/0, 1.0);
  EXPECT_EQ(delay, std::chrono::milliseconds(500));
}

TEST(UtilsTimeTest, RetryJitterFactorWithinExpectedRange) {
  for (int i = 0; i < 10; ++i) {
    double jitter = retry_jitter_factor();
    EXPECT_LE(jitter, 1.0);
    EXPECT_GE(jitter, 0.75);
  }
}

TEST(UtilsTimeTest, SleepForIgnoresNonPositiveDurations) {
  auto start = std::chrono::steady_clock::now();
  sleep_for(std::chrono::milliseconds(0));
  auto end = std::chrono::steady_clock::now();
  EXPECT_LE(end - start, std::chrono::milliseconds(5));
}
