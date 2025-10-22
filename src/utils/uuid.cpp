#include "openai/utils/uuid.hpp"

#include <array>
#include <iomanip>
#include <random>
#include <sstream>

namespace openai::utils {
namespace {

unsigned char random_byte() {
  static thread_local std::random_device rd;
  return static_cast<unsigned char>(rd() & 0xFF);
}

}  // namespace

std::string uuid4() {
  std::array<unsigned char, 16> data{};
  for (auto& byte : data) {
    byte = random_byte();
  }

  data[6] = (data[6] & 0x0F) | 0x40;  // version 4
  data[8] = (data[8] & 0x3F) | 0x80;  // variant 1

  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (std::size_t i = 0; i < data.size(); ++i) {
    oss << std::setw(2) << static_cast<int>(data[i]);
    if (i == 3 || i == 5 || i == 7 || i == 9) {
      oss << '-';
    }
  }
  return oss.str();
}

}  // namespace openai::utils

