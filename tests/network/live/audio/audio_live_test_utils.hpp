#pragma once

#include "network/live/live_test_utils.hpp"

#include "openai/audio.hpp"
#include "openai/files.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace openai::test::live::audio {

namespace detail {

inline void write_little_endian(std::ofstream& out, std::uint32_t value, int byte_count) {
  for (int i = 0; i < byte_count; ++i) {
    out.put(static_cast<char>(value & 0xFF));
    value >>= 8;
  }
}

}  // namespace detail

class TempWavFile {
public:
  explicit TempWavFile(std::string label, double frequency_hz = 440.0, double duration_seconds = 0.8)
      : label_(std::move(label)) {
    const std::filesystem::path base = std::filesystem::temp_directory_path();
    path_ = base / (label_ + "-" + openai::test::live::unique_tag() + ".wav");
    write_wave(frequency_hz, duration_seconds);
  }

  TempWavFile(const TempWavFile&) = delete;
  TempWavFile& operator=(const TempWavFile&) = delete;

  TempWavFile(TempWavFile&& other) noexcept : label_(std::move(other.label_)), path_(std::move(other.path_)), valid_(other.valid_) {
    other.valid_ = false;
  }

  TempWavFile& operator=(TempWavFile&& other) noexcept {
    if (this != &other) {
      cleanup();
      label_ = std::move(other.label_);
      path_ = std::move(other.path_);
      valid_ = other.valid_;
      other.valid_ = false;
    }
    return *this;
  }

  ~TempWavFile() { cleanup(); }

  const std::filesystem::path& path() const { return path_; }

private:
  void write_wave(double frequency_hz, double duration_seconds) {
    constexpr int sample_rate = 16000;
    constexpr int bits_per_sample = 16;
    constexpr int channels = 1;
    constexpr double pi = 3.14159265358979323846;
    constexpr int amplitude = 28000;

    std::size_t sample_count = static_cast<std::size_t>(std::max(duration_seconds, 0.1) * sample_rate);
    if (sample_count == 0) {
      sample_count = static_cast<std::size_t>(0.1 * sample_rate);
    }

    std::vector<std::int16_t> samples(sample_count);
    for (std::size_t i = 0; i < sample_count; ++i) {
      double t = static_cast<double>(i) / sample_rate;
      double value = std::sin(2.0 * pi * frequency_hz * t);
      value = std::clamp(value, -1.0, 1.0);
      samples[i] = static_cast<std::int16_t>(value * amplitude);
    }

    std::ofstream out(path_, std::ios::binary);
    if (!out) {
      throw std::runtime_error("Failed to create temporary WAV file for live audio test.");
    }

    const std::uint32_t data_chunk_size = static_cast<std::uint32_t>(samples.size() * sizeof(std::int16_t));
    const std::uint32_t fmt_chunk_size = 16;
    const std::uint16_t audio_format = 1;
    const std::uint16_t num_channels = channels;
    const std::uint32_t sample_rate_u32 = sample_rate;
    const std::uint16_t block_align = static_cast<std::uint16_t>(num_channels * (bits_per_sample / 8));
    const std::uint32_t byte_rate = sample_rate_u32 * block_align;
    const std::uint32_t riff_chunk_size = 36 + data_chunk_size;

    out.write("RIFF", 4);
    detail::write_little_endian(out, riff_chunk_size, 4);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    detail::write_little_endian(out, fmt_chunk_size, 4);
    detail::write_little_endian(out, audio_format, 2);
    detail::write_little_endian(out, num_channels, 2);
    detail::write_little_endian(out, sample_rate_u32, 4);
    detail::write_little_endian(out, byte_rate, 4);
    detail::write_little_endian(out, block_align, 2);
    detail::write_little_endian(out, bits_per_sample, 2);
    out.write("data", 4);
    detail::write_little_endian(out, data_chunk_size, 4);
    out.write(reinterpret_cast<const char*>(samples.data()), static_cast<std::streamsize>(data_chunk_size));

    if (!out) {
      throw std::runtime_error("Failed to write audio payload for live audio test.");
    }

    valid_ = true;
  }

  void cleanup() {
    if (valid_) {
      std::error_code ec;
      std::filesystem::remove(path_, ec);
      valid_ = false;
    }
  }

  std::string label_;
  std::filesystem::path path_;
  bool valid_ = false;
};

class TempBinaryFile {
public:
  TempBinaryFile(std::string label, std::string extension, const std::vector<std::uint8_t>& data)
      : label_(std::move(label)), extension_(std::move(extension)) {
    if (extension_.empty() || extension_.front() != '.') {
      throw std::invalid_argument("TempBinaryFile extension must begin with '.'");
    }
    const std::filesystem::path base = std::filesystem::temp_directory_path();
    path_ = base / (label_ + "-" + openai::test::live::unique_tag() + extension_);
    std::ofstream out(path_, std::ios::binary);
    if (!out) {
      throw std::runtime_error("Failed to create temporary file for live audio test.");
    }
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!out) {
      throw std::runtime_error("Failed to write audio data for live audio test.");
    }
    valid_ = true;
  }

  TempBinaryFile(const TempBinaryFile&) = delete;
  TempBinaryFile& operator=(const TempBinaryFile&) = delete;

  TempBinaryFile(TempBinaryFile&& other) noexcept
      : label_(std::move(other.label_)), extension_(std::move(other.extension_)), path_(std::move(other.path_)),
        valid_(other.valid_) {
    other.valid_ = false;
  }

  TempBinaryFile& operator=(TempBinaryFile&& other) noexcept {
    if (this != &other) {
      cleanup();
      label_ = std::move(other.label_);
      extension_ = std::move(other.extension_);
      path_ = std::move(other.path_);
      valid_ = other.valid_;
      other.valid_ = false;
    }
    return *this;
  }

  ~TempBinaryFile() { cleanup(); }

  const std::filesystem::path& path() const { return path_; }

private:
  void cleanup() {
    if (valid_) {
      std::error_code ec;
      std::filesystem::remove(path_, ec);
      valid_ = false;
    }
  }

  std::string label_;
  std::string extension_;
  std::filesystem::path path_;
  bool valid_ = false;
};

inline openai::FileUploadRequest make_audio_upload(const std::filesystem::path& path,
                                                   const std::string& content_type) {
  openai::FileUploadRequest upload;
  upload.file_path = path.string();
  upload.file_name = path.filename().string();
  upload.content_type = content_type;
  return upload;
}

inline openai::FileUploadRequest make_wav_upload(const std::filesystem::path& path) {
  return make_audio_upload(path, "audio/wav");
}

inline std::string speech_model() {
  return openai::test::live::get_env("OPENAI_CPP_LIVE_SPEECH_MODEL").value_or("gpt-4o-mini-tts");
}

inline std::string speech_voice() {
  return openai::test::live::get_env("OPENAI_CPP_LIVE_SPEECH_VOICE").value_or("alloy");
}

inline TempBinaryFile synthesize_speech_file(openai::OpenAIClient& client,
                                             const std::string& label,
                                             const std::string& text,
                                             const std::string& format = "wav",
                                             double speed = 1.0) {
  if (format.empty()) {
    throw std::invalid_argument("Speech format must not be empty");
  }

  openai::SpeechRequest request;
  request.input = text;
  request.model = speech_model();
  request.voice = speech_voice();
  request.response_format = format;
  request.speed = speed;

  auto speech = client.audio().speech().create(request);
  if (speech.audio.empty()) {
    throw std::runtime_error("Speech synthesis returned no audio data.");
  }

  std::string extension = "." + format;
  return TempBinaryFile(label, extension, speech.audio);
}

}  // namespace openai::test::live::audio
