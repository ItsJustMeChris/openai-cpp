#pragma once

#include <string>

namespace openai::utils {

/**
 * Generates a random RFC 4122 version 4 UUID using a cryptographically secure
 * random number generator when available.
 */
std::string uuid4();

}  // namespace openai::utils

