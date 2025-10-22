#pragma once

#include <map>
#include <string>

namespace openai::utils {

struct PlatformProperties {
  std::string language;
  std::string package_version;
  std::string os;
  std::string arch;
  std::string runtime;
  std::string runtime_version;
};

/**
 * Returns cached platform properties describing the current runtime environment
 * in the same shape exposed by the OpenAI TypeScript SDK.
 */
const PlatformProperties& platform_properties();

/**
 * Returns the `X-Stainless-*` headers required for telemetry parity with
 * the TypeScript SDK.
 */
const std::map<std::string, std::string>& platform_headers();

/**
 * Returns the default User-Agent string for the SDK.
 */
std::string user_agent();

}  // namespace openai::utils

