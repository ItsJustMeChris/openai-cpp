#include "openai/client.hpp"
#include "openai/images.hpp"
#include "openai/utils/base64.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::filesystem::path executable_directory(const char* argv0) {
  if (argv0 && argv0[0] != '\0') {
    std::filesystem::path candidate = std::filesystem::absolute(std::filesystem::path(argv0));
    try {
      candidate = std::filesystem::canonical(candidate);
      return candidate.parent_path();
    } catch (const std::exception&) {
      try {
        candidate = std::filesystem::weakly_canonical(candidate);
        return candidate.parent_path();
      } catch (const std::exception&) {
      }
    }
  }
  return std::filesystem::current_path();
}

std::string build_prompt(int argc, char** argv) {
  if (argc <= 1) {
    return "A whimsical watercolor illustration of a friendly robot exploring a lush garden at sunrise";
  }

  std::ostringstream joined;
  joined << argv[1];
  for (int i = 2; i < argc; ++i) {
    joined << ' ' << argv[i];
  }
  return joined.str();
}

}  // namespace

int main(int argc, char** argv) {
  const char* api_key = std::getenv("OPENAI_API_KEY");
  if (!api_key) {
    std::cerr << "OPENAI_API_KEY environment variable must be set\n";
    return 1;
  }

  try {
    openai::ClientOptions options;
    options.api_key = api_key;

    openai::OpenAIClient client(options);

    openai::ImageGenerateRequest request;
    request.model = "gpt-image-1";
    request.prompt = build_prompt(argc, argv);
    request.size = "1024x1024";

    std::cout << "Requesting image from gpt-image-1...\n";
    auto response = client.images().generate(request);

    if (response.data.empty() || !response.data.front().b64_json) {
      std::cerr << "Image generation response missing image data\n";
      return 1;
    }

    const std::string& base64_image = *response.data.front().b64_json;
    std::vector<std::uint8_t> image_bytes = openai::utils::decode_base64(base64_image);

    if (image_bytes.empty()) {
      std::cerr << "Decoded image data is empty\n";
      return 1;
    }

    std::filesystem::path output_path = executable_directory(argv[0]) / "gpt_image_demo.png";
    std::ofstream output_file(output_path, std::ios::binary);
    if (!output_file) {
      std::cerr << "Failed to open file for writing: " << output_path << "\n";
      return 1;
    }

    output_file.write(reinterpret_cast<const char*>(image_bytes.data()),
                      static_cast<std::streamsize>(image_bytes.size()));
    if (!output_file) {
      std::cerr << "Failed to write image data to: " << output_path << "\n";
      return 1;
    }

    std::cout << "Image written to " << output_path << "\n";
  } catch (const openai::OpenAIError& error) {
    std::cerr << "OpenAI error: " << error.what() << "\n";
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "Unexpected error: " << error.what() << "\n";
    return 1;
  }

  return 0;
}

