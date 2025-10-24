#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>

#include "openai/audio.hpp"
#include "openai/client.hpp"
#include "openai/responses.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include <TargetConditionals.h>

namespace {

std::string build_prompt(int argc, char *argv[])
{
  if (argc <= 1)
  {
    return "Please introduce yourself and describe one interesting capability of the Responses API.";
  }

  std::ostringstream oss;
  for (int i = 1; i < argc; ++i)
  {
    if (i > 1)
    {
      oss << ' ';
    }
    oss << argv[i];
  }
  return oss.str();
}

void run_audio_demo(int argc, char *argv[])
{
  const char *api_key = std::getenv("OPENAI_API_KEY");
  if (!api_key)
  {
    std::cerr << "OPENAI_API_KEY environment variable must be set\n";
    std::exit(1);
  }

  openai::ClientOptions options;
  options.api_key = api_key;

  std::cout << "Creating OpenAI client...\n";
  openai::OpenAIClient client(options);

  const std::string prompt = build_prompt(argc, argv);
  std::cout << "Prompt: " << prompt << std::endl;

  openai::ResponseRequest response_request;
  response_request.model = "gpt-4o-mini";
  response_request.input = {
      openai::ResponseInputItem{
          .type = openai::ResponseInputItem::Type::Message,
          .message = openai::ResponseInputMessage{
              .role = "user",
              .content = {openai::ResponseInputContent{
                  .type = openai::ResponseInputContent::Type::Text,
                  .text = prompt}}}}};

  std::cout << "Requesting response...\n";
  openai::Response response = client.responses().create(response_request);

  std::string final_text = response.output_text;
  auto append_from_message = [&](const openai::ResponseOutputMessage &message) {
    for (const auto &segment : message.text_segments)
    {
      final_text += segment.text;
    }

    for (const auto &content : message.content)
    {
      if (content.type == openai::ResponseOutputContent::Type::Text && content.text)
      {
        final_text += content.text->text;
      }
    }
  };

  if (final_text.empty())
  {
    for (const auto &item : response.output)
    {
      if (item.message)
      {
        append_from_message(*item.message);
      }
    }
  }

  if (final_text.empty())
  {
    for (const auto &message : response.messages)
    {
      if (message.role == "assistant")
      {
        append_from_message(message);
      }
    }
  }

  if (final_text.empty())
  {
    std::cerr << "No textual output available to synthesize.\n";
    std::exit(1);
  }

  std::cout << "Response text:\n"
            << final_text << std::endl;

  openai::SpeechRequest speech_request;
  speech_request.model = "gpt-4o-mini-tts";
  speech_request.voice = "alloy";
  speech_request.response_format = "aac";
  speech_request.input = final_text;

  std::cout << "Requesting speech synthesis...\n";
  openai::SpeechResponse speech_response = client.audio().speech().create(speech_request);

  if (speech_response.audio.empty())
  {
    std::cerr << "Received empty audio buffer.\n";
    std::exit(1);
  }

  NSData *audio_data = [NSData dataWithBytes:speech_response.audio.data()
                                      length:speech_response.audio.size()];
  if (!audio_data || [audio_data length] == 0)
  {
    std::cerr << "Failed to create NSData from synthesized audio.\n";
    std::exit(1);
  }

  NSError *player_error = nil;
  AVAudioPlayer *player = [[AVAudioPlayer alloc] initWithData:audio_data error:&player_error];
  if (!player)
  {
    const char *error_desc =
        player_error ? [[player_error localizedDescription] UTF8String] : "unknown error";
    std::cerr << "Failed to initialize AVAudioPlayer: " << error_desc << std::endl;
    std::exit(1);
  }

#if TARGET_OS_IPHONE || TARGET_OS_TV || TARGET_OS_WATCH
  AVAudioSession *session = [AVAudioSession sharedInstance];
  NSError *session_error = nil;
  if (![session setCategory:AVAudioSessionCategoryPlayback error:&session_error])
  {
    const char *error_desc =
        session_error ? [[session_error localizedDescription] UTF8String] : "unknown error";
    std::cerr << "Failed to set AVAudioSession category: " << error_desc << std::endl;
  }
  if (![session setActive:YES error:&session_error])
  {
    const char *error_desc =
        session_error ? [[session_error localizedDescription] UTF8String] : "unknown error";
    std::cerr << "Failed to activate AVAudioSession: " << error_desc << std::endl;
  }
#endif

  if (![player prepareToPlay])
  {
    std::cerr << "AVAudioPlayer failed to prepare audio playback.\n";
    std::exit(1);
  }

  std::cout << "Playing audio...\n";
  if (![player play])
  {
    std::cerr << "AVAudioPlayer failed to start playback.\n";
    std::exit(1);
  }

  while ([player isPlaying])
  {
    [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.1]];
  }

  std::cout << "Playback complete.\n";
}

} // namespace

int main(int argc, char *argv[])
{
  @autoreleasepool
  {
    try
    {
      run_audio_demo(argc, argv);
    }
    catch (const openai::OpenAIError &error)
    {
      std::cerr << "OpenAI error: " << error.what() << std::endl;
      return 1;
    }
    catch (const std::exception &error)
    {
      std::cerr << "Unexpected error: " << error.what() << std::endl;
      return 1;
    }
  }

  return 0;
}
