#include "openai/utils/platform.hpp"

#include <sstream>
#include <string>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

namespace openai::utils {
namespace {

#ifdef OPENAI_CPP_VERSION
constexpr const char* kPackageVersion = OPENAI_CPP_VERSION;
#else
constexpr const char* kPackageVersion = "0.0.0-dev";
#endif

std::string detect_os() {
#if defined(__APPLE__) && defined(__MACH__)
#if defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
  return "iOS";
#elif defined(TARGET_OS_MAC) && TARGET_OS_MAC
  return "MacOS";
#else
  return "Other:apple";
#endif
#elif defined(__ANDROID__)
  return "Android";
#elif defined(_WIN32)
  return "Windows";
#elif defined(__linux__)
  return "Linux";
#elif defined(__FreeBSD__)
  return "FreeBSD";
#elif defined(__OpenBSD__)
  return "OpenBSD";
#elif defined(__EMSCRIPTEN__)
  return "Other:emscripten";
#elif defined(__unix__)
  return "Other:unix";
#else
  return "Unknown";
#endif
}

std::string detect_arch() {
#if defined(__x86_64__) || defined(_M_X64)
  return "x64";
#elif defined(__i386__) || defined(_M_IX86)
  return "x32";
#elif defined(__aarch64__) || defined(_M_ARM64)
  return "arm64";
#elif defined(__arm__) || defined(_M_ARM)
  return "arm";
#elif defined(__EMSCRIPTEN__)
  return "other:wasm32";
#elif defined(__ppc64__) || defined(__powerpc64__) || defined(_M_PPC)
  return "other:ppc64";
#elif defined(__ppc__) || defined(__powerpc__)
  return "other:ppc";
#elif defined(__mips__) || defined(__mips) || defined(_M_MRX000)
  return "other:mips";
#else
  return "unknown";
#endif
}

std::string detect_runtime_version() {
#if defined(__clang__)
  std::ostringstream oss;
  oss << __clang_major__ << '.' << __clang_minor__ << '.' << __clang_patchlevel__;
  return oss.str();
#elif defined(_MSC_VER)
  std::ostringstream oss;
  oss << (_MSC_VER / 100) << '.' << (_MSC_VER % 100);
#if defined(_MSC_FULL_VER)
  oss << '.' << (_MSC_FULL_VER % 100000);
#endif
  return oss.str();
#elif defined(__GNUC__)
  std::ostringstream oss;
  oss << __GNUC__ << '.' << __GNUC_MINOR__ << '.' << __GNUC_PATCHLEVEL__;
  return oss.str();
#elif defined(__VERSION__)
  return std::string(__VERSION__);
#else
  return "unknown";
#endif
}

PlatformProperties compute_properties() {
  PlatformProperties props;
  props.language = "cpp";
  props.package_version = kPackageVersion;
  props.os = detect_os();
  props.arch = detect_arch();
  props.runtime = "cpp";
  props.runtime_version = detect_runtime_version();
  return props;
}

std::map<std::string, std::string> build_headers(const PlatformProperties& props) {
  return {
      {"X-Stainless-Lang", props.language},
      {"X-Stainless-Package-Version", props.package_version},
      {"X-Stainless-OS", props.os},
      {"X-Stainless-Arch", props.arch},
      {"X-Stainless-Runtime", props.runtime},
      {"X-Stainless-Runtime-Version", props.runtime_version},
  };
}

}  // namespace

const PlatformProperties& platform_properties() {
  static const PlatformProperties props = compute_properties();
  return props;
}

const std::map<std::string, std::string>& platform_headers() {
  static const std::map<std::string, std::string> headers = build_headers(platform_properties());
  return headers;
}

std::string user_agent() {
  const auto& props = platform_properties();
  return std::string("OpenAI/CPP ") + props.package_version;
}

}  // namespace openai::utils
