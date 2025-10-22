#include <gtest/gtest.h>

#include <chrono>
#include <map>
#include <string>
#include <vector>

#include "openai/client.hpp"
#include "openai/webhooks.hpp"
#include "support/mock_http_client.hpp"

namespace {

std::string base64_encode(const std::vector<std::uint8_t>& data) {
  static constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  std::size_t i = 0;
  while (i + 2 < data.size()) {
    std::uint32_t triple = (static_cast<std::uint32_t>(data[i]) << 16) |
                           (static_cast<std::uint32_t>(data[i + 1]) << 8) |
                           static_cast<std::uint32_t>(data[i + 2]);
    out.push_back(alphabet[(triple >> 18) & 0x3F]);
    out.push_back(alphabet[(triple >> 12) & 0x3F]);
    out.push_back(alphabet[(triple >> 6) & 0x3F]);
    out.push_back(alphabet[triple & 0x3F]);
    i += 3;
  }
  if (i < data.size()) {
    std::uint32_t triple = static_cast<std::uint32_t>(data[i]) << 16;
    out.push_back(alphabet[(triple >> 18) & 0x3F]);
    if (i + 1 < data.size()) {
      triple |= static_cast<std::uint32_t>(data[i + 1]) << 8;
      out.push_back(alphabet[(triple >> 12) & 0x3F]);
      out.push_back(alphabet[(triple >> 6) & 0x3F]);
      out.push_back('=');
    } else {
      out.push_back(alphabet[(triple >> 12) & 0x3F]);
      out.push_back('=');
      out.push_back('=');
    }
  }
  return out;
}

class HMACSHA256 {
public:
  static std::vector<std::uint8_t> compute(const std::string& key, const std::string& message) {
    std::vector<std::uint8_t> k(key.begin(), key.end());
    const std::size_t block_size = 64;
    if (k.size() > block_size) {
      k = sha256(k);
    }
    k.resize(block_size, 0x00);

    std::vector<std::uint8_t> o_key_pad(block_size);
    std::vector<std::uint8_t> i_key_pad(block_size);
    for (std::size_t i = 0; i < block_size; ++i) {
      o_key_pad[i] = static_cast<std::uint8_t>(k[i] ^ 0x5c);
      i_key_pad[i] = static_cast<std::uint8_t>(k[i] ^ 0x36);
    }

    std::vector<std::uint8_t> inner(i_key_pad);
    inner.insert(inner.end(), message.begin(), message.end());
    auto inner_hash = sha256(inner);

    std::vector<std::uint8_t> outer(o_key_pad);
    outer.insert(outer.end(), inner_hash.begin(), inner_hash.end());
    return sha256(outer);
  }

private:
  static std::vector<std::uint8_t> sha256(const std::vector<std::uint8_t>& data) {
    struct State {
      std::uint32_t h[8];
      std::uint64_t bits = 0;
      std::uint8_t buffer[64];
      std::size_t filled = 0;

      State() { reset(); }

      void reset() {
        std::uint32_t init[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
        std::copy(std::begin(init), std::end(init), h);
        bits = 0;
        filled = 0;
      }

      void update(const std::uint8_t* chunk, std::size_t len) {
        while (len > 0) {
          std::size_t copy_len = std::min<std::size_t>(len, 64 - filled);
          std::copy(chunk, chunk + copy_len, buffer + filled);
          filled += copy_len;
          chunk += copy_len;
          len -= copy_len;
          if (filled == 64) {
            process(buffer);
            bits += 512;
            filled = 0;
          }
        }
      }

      void finalize(std::vector<std::uint8_t>& out) {
        bits += filled * 8;
        buffer[filled++] = 0x80;
        if (filled > 56) {
          while (filled < 64) buffer[filled++] = 0;
          process(buffer);
          filled = 0;
        }
        while (filled < 56) buffer[filled++] = 0;
        for (int i = 7; i >= 0; --i) {
          buffer[filled++] = static_cast<std::uint8_t>((bits >> (i * 8)) & 0xff);
        }
        process(buffer);
        out.resize(32);
        for (int i = 0; i < 8; ++i) {
          out[i * 4] = static_cast<std::uint8_t>((h[i] >> 24) & 0xff);
          out[i * 4 + 1] = static_cast<std::uint8_t>((h[i] >> 16) & 0xff);
          out[i * 4 + 2] = static_cast<std::uint8_t>((h[i] >> 8) & 0xff);
          out[i * 4 + 3] = static_cast<std::uint8_t>(h[i] & 0xff);
        }
      }

      void process(const std::uint8_t* chunk) {
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
        auto load32 = [](const std::uint8_t* p) {
          return (static_cast<std::uint32_t>(p[0]) << 24) |
                 (static_cast<std::uint32_t>(p[1]) << 16) |
                 (static_cast<std::uint32_t>(p[2]) << 8) |
                 static_cast<std::uint32_t>(p[3]);
        };
        auto rotr = [](std::uint32_t x, std::uint32_t n) {
          return (x >> n) | (x << (32 - n));
        };

        std::uint32_t w[64];
        for (int t = 0; t < 16; ++t) w[t] = load32(chunk + t * 4);
        for (int t = 16; t < 64; ++t) {
          std::uint32_t s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
          std::uint32_t s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
          w[t] = w[t - 16] + s0 + w[t - 7] + s1;
        }

        std::uint32_t a = h[0];
        std::uint32_t b = h[1];
        std::uint32_t c = h[2];
        std::uint32_t d = h[3];
        std::uint32_t e = h[4];
        std::uint32_t f = h[5];
        std::uint32_t g = h[6];
        std::uint32_t h_val = h[7];
        for (int t = 0; t < 64; ++t) {
          std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
          std::uint32_t ch = (e & f) ^ ((~e) & g);
          std::uint32_t temp1 = h_val + S1 + ch + k[t] + w[t];
          std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
          std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
          std::uint32_t temp2 = S0 + maj;
          h_val = g;
          g = f;
          f = e;
          e = d + temp1;
          d = c;
          c = b;
          b = a;
          a = temp1 + temp2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += h_val;
      }
    } state;

    state.update(data.data(), data.size());
    std::vector<std::uint8_t> digest;
    state.finalize(digest);
    return digest;
  }
};

std::string build_signature(const std::string& secret,
                            const std::string& payload,
                            const std::string& timestamp,
                            const std::string& webhook_id) {
  std::string signed_payload = webhook_id.empty() ? (timestamp + "." + payload)
                                                  : (webhook_id + "." + timestamp + "." + payload);
  auto digest = HMACSHA256::compute(secret, signed_payload);
  return base64_encode(digest);
}

}  // namespace

TEST(WebhooksResourceTest, VerifySignatureAndUnwrap) {
  using namespace openai;

  auto http = std::make_unique<openai::testing::MockHttpClient>();
  ClientOptions options;
  options.api_key = "sk-test";
  options.webhook_secret = "whsec_bXlzZWNyZXQ=";

  OpenAIClient client(std::move(options), std::move(http));

  nlohmann::json payload = {
      {"id", "evt_123"},
      {"created_at", 1700000000},
      {"object", "event"},
      {"type", "response.completed"},
      {"data", {{"id", "resp_123"}}}};
  std::string payload_str = payload.dump();

  auto timestamp = std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
                                           std::chrono::system_clock::now().time_since_epoch())
                                       .count());
  std::string webhook_id = "wh_abc";
  std::string secret = "mysecret";
  auto signature = build_signature(secret, payload_str, timestamp, webhook_id);

  std::map<std::string, std::string> headers = {
      {"webhook-signature", "v1," + signature},
      {"webhook-timestamp", timestamp},
      {"webhook-id", webhook_id}};

  auto event = client.webhooks().unwrap(payload_str, headers);
  EXPECT_EQ(event.id, "evt_123");
  EXPECT_EQ(event.created_at, 1700000000);
  EXPECT_EQ(event.object, "event");
  EXPECT_EQ(event.type, openai::webhooks::EventType::ResponseCompleted);

  const auto* response_data = std::get_if<openai::webhooks::ResponseEventData>(&event.data);
  ASSERT_NE(response_data, nullptr);
  EXPECT_EQ(response_data->id, "resp_123");
}

TEST(WebhooksResourceTest, InvalidSignatureFails) {
  using namespace openai;

  auto http = std::make_unique<openai::testing::MockHttpClient>();
  ClientOptions options;
  options.api_key = "sk-test";
  options.webhook_secret = "whsec_bXlzZWNyZXQ=";

  OpenAIClient client(std::move(options), std::move(http));

  std::string payload = R"({"id":"evt","created_at":1,"object":"event","type":"batch.cancelled","data":{"id":"batch_1"}})";
  std::map<std::string, std::string> headers = {
      {"webhook-signature", "v1,invalid"},
      {"webhook-timestamp", std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
                                     std::chrono::system_clock::now().time_since_epoch())
                                     .count())}};

  EXPECT_FALSE(client.webhooks().verify_signature(payload, headers, {}));
  EXPECT_THROW(client.webhooks().unwrap(payload, headers), openai::OpenAIError);
}
