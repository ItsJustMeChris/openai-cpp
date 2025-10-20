#include "openai/utils/base64.hpp"

#include "openai/error.hpp"

#include <array>
#include <cctype>
#include <stdexcept>
#include <string_view>

namespace openai::utils {
namespace {

const std::array<int, 256>& decode_table() {
  static const std::array<int, 256> table = [] {
    std::array<int, 256> t{};
    t.fill(-1);
    const std::string_view alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (std::size_t i = 0; i < alphabet.size(); ++i) {
      t[static_cast<unsigned char>(alphabet[i])] = static_cast<int>(i);
    }
    return t;
  }();
  return table;
}

int decode_char(unsigned char c) {
  int value = decode_table()[c];
  if (value == -1) {
    throw OpenAIError("Invalid base64 character encountered");
  }
  return value;
}

}  // namespace

std::vector<std::uint8_t> decode_base64(std::string_view input) {
  std::string_view trimmed = input;
  while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.front()))) {
    trimmed.remove_prefix(1);
  }
  while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.back()))) {
    trimmed.remove_suffix(1);
  }

  if (trimmed.size() % 4 != 0) {
    throw OpenAIError("Base64 input length must be a multiple of 4");
  }

  std::size_t padding = 0;
  if (!trimmed.empty() && trimmed.back() == '=') {
    padding = 1;
    if (trimmed.size() >= 2 && trimmed[trimmed.size() - 2] == '=') {
      padding = 2;
    }
  }

  std::size_t output_size = (trimmed.size() / 4) * 3 - padding;
  std::vector<std::uint8_t> output;
  output.reserve(output_size);

  for (std::size_t i = 0; i < trimmed.size(); i += 4) {
    int sextet_a = trimmed[i] == '=' ? 0 : decode_char(static_cast<unsigned char>(trimmed[i]));
    int sextet_b = trimmed[i + 1] == '=' ? 0 : decode_char(static_cast<unsigned char>(trimmed[i + 1]));
    int sextet_c = trimmed[i + 2] == '=' ? 0 : decode_char(static_cast<unsigned char>(trimmed[i + 2]));
    int sextet_d = trimmed[i + 3] == '=' ? 0 : decode_char(static_cast<unsigned char>(trimmed[i + 3]));

    std::uint32_t triple = (static_cast<std::uint32_t>(sextet_a) << 18) |
                           (static_cast<std::uint32_t>(sextet_b) << 12) |
                           (static_cast<std::uint32_t>(sextet_c) << 6) |
                           static_cast<std::uint32_t>(sextet_d);

    output.push_back(static_cast<std::uint8_t>((triple >> 16) & 0xFF));
    if (trimmed[i + 2] != '=') {
      output.push_back(static_cast<std::uint8_t>((triple >> 8) & 0xFF));
    }
    if (trimmed[i + 3] != '=') {
      output.push_back(static_cast<std::uint8_t>(triple & 0xFF));
    }
  }

  return output;
}

}  // namespace openai::utils
