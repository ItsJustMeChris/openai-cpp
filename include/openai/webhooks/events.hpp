#pragma once

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

namespace openai::webhooks {

struct BatchEventData {
  std::string id;
};

struct EvalRunEventData {
  std::string id;
};

struct FineTuningJobEventData {
  std::string id;
};

struct RealtimeCallIncomingData {
  std::string call_id;
  struct SipHeader {
    std::string name;
    std::string value;
  };
  std::vector<SipHeader> sip_headers;
};

struct ResponseEventData {
  std::string id;
};

enum class EventType {
  BatchCancelled,
  BatchCompleted,
  BatchExpired,
  BatchFailed,
  EvalRunCanceled,
  EvalRunFailed,
  EvalRunSucceeded,
  FineTuningJobCancelled,
  FineTuningJobFailed,
  FineTuningJobSucceeded,
  RealtimeCallIncoming,
  ResponseCancelled,
  ResponseCompleted,
  ResponseFailed,
  ResponseIncomplete,
  Unknown
};

using EventData = std::variant<std::monostate,
                               BatchEventData,
                               EvalRunEventData,
                               FineTuningJobEventData,
                               RealtimeCallIncomingData,
                               ResponseEventData>;

struct WebhookEvent {
  std::string id;
  int created_at = 0;
  std::string object;
  EventType type = EventType::Unknown;
  EventData data;
  nlohmann::json raw = nlohmann::json::object();
};

EventType parse_event_type(const std::string& type);

}  // namespace openai::webhooks
