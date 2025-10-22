#include "openai/webhooks.hpp"

#include "openai/client.hpp"
#include "openai/error.hpp"
#include "openai/utils/base64.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace openai {
namespace {

class SHA256 {
public:
  SHA256() { reset(); }

  void reset() {
    state_[0] = 0x6a09e667u;
    state_[1] = 0xbb67ae85u;
    state_[2] = 0x3c6ef372u;
    state_[3] = 0xa54ff53au;
    state_[4] = 0x510e527fu;
    state_[5] = 0x9b05688cu;
    state_[6] = 0x1f83d9abu;
    state_[7] = 0x5be0cd19u;
    bit_length_ = 0;
    buffer_size_ = 0;
  }

  void update(const std::uint8_t* data, std::size_t size) {
    while (size > 0) {
      std::size_t to_copy = std::min<std::size_t>(size, 64 - buffer_size_);
      std::copy(data, data + to_copy, buffer_.begin() + buffer_size_);
      buffer_size_ += to_copy;
      data += to_copy;
      size -= to_copy;
      if (buffer_size_ == 64) {
        process_block(buffer_.data());
        bit_length_ += 512;
        buffer_size_ = 0;
      }
    }
  }

  std::vector<std::uint8_t> digest() {
    bit_length_ += static_cast<std::uint64_t>(buffer_size_) * 8;
    buffer_[buffer_size_++] = 0x80u;
    if (buffer_size_ > 56) {
      while (buffer_size_ < 64) buffer_[buffer_size_++] = 0;
      process_block(buffer_.data());
      buffer_size_ = 0;
    }
    while (buffer_size_ < 56) buffer_[buffer_size_++] = 0;
    for (int i = 7; i >= 0; --i) {
      buffer_[buffer_size_++] = static_cast<std::uint8_t>((bit_length_ >> (i * 8)) & 0xffu);
    }
    process_block(buffer_.data());

    std::vector<std::uint8_t> out(32);
    for (int i = 0; i < 8; ++i) {
      out[i * 4] = static_cast<std::uint8_t>((state_[i] >> 24) & 0xffu);
      out[i * 4 + 1] = static_cast<std::uint8_t>((state_[i] >> 16) & 0xffu);
      out[i * 4 + 2] = static_cast<std::uint8_t>((state_[i] >> 8) & 0xffu);
      out[i * 4 + 3] = static_cast<std::uint8_t>(state_[i] & 0xffu);
    }
    return out;
  }

private:
  static std::uint32_t rotr(std::uint32_t x, std::uint32_t n) {
    return (x >> n) | (x << (32 - n));
  }

  static std::uint32_t load32(const std::uint8_t* p) {
    return (static_cast<std::uint32_t>(p[0]) << 24) |
           (static_cast<std::uint32_t>(p[1]) << 16) |
           (static_cast<std::uint32_t>(p[2]) << 8) |
           static_cast<std::uint32_t>(p[3]);
  }

  void process_block(const std::uint8_t* block) {
    static constexpr std::uint32_t k[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
        0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
        0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
        0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
        0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
        0xc67178f2u};

    std::uint32_t w[64];
    for (int t = 0; t < 16; ++t) w[t] = load32(block + t * 4);
    for (int t = 16; t < 64; ++t) {
      std::uint32_t s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
      std::uint32_t s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
      w[t] = w[t - 16] + s0 + w[t - 7] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];

    for (int t = 0; t < 64; ++t) {
      std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      std::uint32_t ch = (e & f) ^ ((~e) & g);
      std::uint32_t temp1 = h + S1 + ch + k[t] + w[t];
      std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      std::uint32_t temp2 = S0 + maj;

      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }

    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::uint32_t state_[8]{};
  std::uint64_t bit_length_ = 0;
  std::size_t buffer_size_ = 0;
  std::array<std::uint8_t, 64> buffer_{};
};

std::vector<std::uint8_t> sha256(const std::string& data) {
  SHA256 ctx;
  ctx.update(reinterpret_cast<const std::uint8_t*>(data.data()), data.size());
  return ctx.digest();
}

std::vector<std::uint8_t> hmac_sha256(const std::string& key, const std::string& message) {
  constexpr std::size_t block_size = 64;
  std::vector<std::uint8_t> k(key.begin(), key.end());
  if (k.size() > block_size) {
    k = sha256(key);
  }
  k.resize(block_size, 0x00);

  std::vector<std::uint8_t> o_key_pad(block_size);
  std::vector<std::uint8_t> i_key_pad(block_size);
  for (std::size_t i = 0; i < block_size; ++i) {
    o_key_pad[i] = static_cast<std::uint8_t>(k[i] ^ 0x5c);
    i_key_pad[i] = static_cast<std::uint8_t>(k[i] ^ 0x36);
  }

  std::string inner_msg;
  inner_msg.reserve(block_size + message.size());
  inner_msg.assign(reinterpret_cast<const char*>(i_key_pad.data()), block_size);
  inner_msg.append(message);
  auto inner_hash = sha256(inner_msg);

  std::string outer_msg;
  outer_msg.reserve(block_size + inner_hash.size());
  outer_msg.assign(reinterpret_cast<const char*>(o_key_pad.data()), block_size);
  outer_msg.append(reinterpret_cast<const char*>(inner_hash.data()), inner_hash.size());
  return sha256(outer_msg);
}

std::string trim(const std::string& input) {
  std::size_t start = 0;
  while (start < input.size() && std::isspace(static_cast<unsigned char>(input[start]))) ++start;
  std::size_t end = input.size();
  while (end > start && std::isspace(static_cast<unsigned char>(input[end - 1]))) --end;
  return input.substr(start, end - start);
}

std::vector<std::string> split_signatures(const std::string& header) {
  std::vector<std::string> parts;
  std::stringstream ss(header);
  std::string item;
  while (std::getline(ss, item, ' ')) {
    auto trimmed = trim(item);
    if (trimmed.rfind("v1,", 0) == 0) {
      parts.push_back(trimmed.substr(3));
    } else if (!trimmed.empty()) {
      parts.push_back(trimmed);
    }
  }
  return parts;
}

bool timing_safe_equals(const std::vector<std::uint8_t>& lhs, const std::vector<std::uint8_t>& rhs) {
  if (lhs.size() != rhs.size()) return false;
  std::uint8_t value = 0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    value |= static_cast<std::uint8_t>(lhs[i] ^ rhs[i]);
  }
  return value == 0;
}

std::string get_required_header(const std::map<std::string, std::string>& headers, const std::string& key) {
  auto it = headers.find(key);
  if (it == headers.end()) {
    throw OpenAIError("Missing required header: " + key);
  }
  return it->second;
}

std::string normalize_secret(const std::string& secret) {
  if (secret.rfind("whsec_", 0) == 0) {
    auto decoded = utils::decode_base64(secret.substr(6));
    return std::string(decoded.begin(), decoded.end());
  }
  return secret;
}

webhooks::EventType map_event_type(const std::string& type) {
  using webhooks::EventType;
  if (type == "batch.cancelled") return EventType::BatchCancelled;
  if (type == "batch.completed") return EventType::BatchCompleted;
  if (type == "batch.expired") return EventType::BatchExpired;
  if (type == "batch.failed") return EventType::BatchFailed;
  if (type == "eval.run.canceled") return EventType::EvalRunCanceled;
  if (type == "eval.run.failed") return EventType::EvalRunFailed;
  if (type == "eval.run.succeeded") return EventType::EvalRunSucceeded;
  if (type == "fine_tuning.job.cancelled") return EventType::FineTuningJobCancelled;
  if (type == "fine_tuning.job.failed") return EventType::FineTuningJobFailed;
  if (type == "fine_tuning.job.succeeded") return EventType::FineTuningJobSucceeded;
  if (type == "realtime.call.incoming") return EventType::RealtimeCallIncoming;
  if (type == "response.cancelled") return EventType::ResponseCancelled;
  if (type == "response.completed") return EventType::ResponseCompleted;
  if (type == "response.failed") return EventType::ResponseFailed;
  if (type == "response.incomplete") return EventType::ResponseIncomplete;
  return EventType::Unknown;
}

webhooks::EventData parse_event_data(webhooks::EventType type, const nlohmann::json& data_json) {
  using namespace webhooks;
  if (!data_json.is_object()) {
    return std::monostate{};
  }
  switch (type) {
    case EventType::BatchCancelled:
    case EventType::BatchCompleted:
    case EventType::BatchExpired:
    case EventType::BatchFailed: {
      BatchEventData data;
      data.id = data_json.value("id", "");
      return data;
    }
    case EventType::EvalRunCanceled:
    case EventType::EvalRunFailed:
    case EventType::EvalRunSucceeded: {
      EvalRunEventData data;
      data.id = data_json.value("id", "");
      return data;
    }
    case EventType::FineTuningJobCancelled:
    case EventType::FineTuningJobFailed:
    case EventType::FineTuningJobSucceeded: {
      FineTuningJobEventData data;
      data.id = data_json.value("id", "");
      return data;
    }
    case EventType::ResponseCancelled:
    case EventType::ResponseCompleted:
    case EventType::ResponseFailed:
    case EventType::ResponseIncomplete: {
      ResponseEventData data;
      data.id = data_json.value("id", "");
      return data;
    }
    case EventType::RealtimeCallIncoming: {
      RealtimeCallIncomingData data;
      data.id = data_json.value("id", "");
      data.session_id = data_json.value("session_id", "");
      data.call_id = data_json.value("call_id", "");
      if (data_json.contains("sip_headers") && data_json.at("sip_headers").is_array()) {
        for (const auto& header_json : data_json.at("sip_headers")) {
          RealtimeCallIncomingData::SipHeader header;
          header.name = header_json.value("name", "");
          header.value = header_json.value("value", "");
          data.sip_headers.push_back(std::move(header));
        }
      }
      return data;
    }
    default:
      return std::monostate{};
  }
}

}  // namespace

namespace webhooks {

EventType parse_event_type(const std::string& type) {
  return ::openai::map_event_type(type);
}

}  // namespace webhooks

bool WebhooksResource::verify_signature(const std::string& payload,
                                        const std::map<std::string, std::string>& headers,
                                        const WebhookVerifyOptions& options) const {
  const std::string signature_header = get_required_header(headers, "webhook-signature");
  const std::string timestamp_header = get_required_header(headers, "webhook-timestamp");
  std::string webhook_id;
  auto it = headers.find("webhook-id");
  if (it != headers.end()) {
    webhook_id = it->second;
  }

  auto timestamp_secs = std::stoll(timestamp_header);
  auto now_secs = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::system_clock::now().time_since_epoch());
  auto timestamp = std::chrono::seconds(timestamp_secs);
  if (now_secs - timestamp > options.tolerance) return false;
  if (timestamp > now_secs + options.tolerance) return false;

  std::string secret;
  if (options.secret) {
    secret = *options.secret;
  } else if (client_.options().webhook_secret) {
    secret = *client_.options().webhook_secret;
  }
  if (secret.empty()) {
    throw OpenAIError("Webhook secret must be provided for verification");
  }
  secret = normalize_secret(secret);

  auto signatures = split_signatures(signature_header);
  if (signatures.empty()) return false;

  std::string signed_payload;
  if (!webhook_id.empty()) {
    signed_payload = webhook_id + "." + timestamp_header + "." + payload;
  } else {
    signed_payload = timestamp_header + "." + payload;
  }
  auto expected = hmac_sha256(secret, signed_payload);

  for (const auto& signature : signatures) {
    std::vector<std::uint8_t> decoded;
    try {
      decoded = utils::decode_base64(signature);
    } catch (const std::exception&) {
      continue;
    }
    if (timing_safe_equals(decoded, expected)) {
      return true;
    }
  }

  return false;
}

webhooks::WebhookEvent WebhooksResource::unwrap(const std::string& payload,
                                                const std::map<std::string, std::string>& headers,
                                                const WebhookVerifyOptions& options) const {
  if (!verify_signature(payload, headers, options)) {
    throw OpenAIError("Invalid webhook signature");
  }

  auto parsed = nlohmann::json::parse(payload);
  webhooks::WebhookEvent event;
  event.raw = parsed;
  event.id = parsed.value("id", "");
  event.created_at = parsed.value("created_at", 0);
  event.object = parsed.value("object", "");
  auto type_string = parsed.value("type", "");
  event.type = map_event_type(type_string);
  if (parsed.contains("data")) {
    event.data = parse_event_data(event.type, parsed.at("data"));
  }
  return event;
}

}  // namespace openai
