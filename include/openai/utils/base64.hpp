#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace openai::utils {

std::vector<std::uint8_t> decode_base64(std::string_view input);

}

