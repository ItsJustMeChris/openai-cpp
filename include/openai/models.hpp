#pragma once

#include <string>
#include <vector>

namespace openai {

struct Model {
  std::string id;
  int created = 0;
  std::string object;
  std::string owned_by;
};

struct ModelDeleted {
  std::string id;
  bool deleted = false;
  std::string object;
};

struct ModelList {
  std::string object;
  std::vector<Model> data;
};

}  // namespace openai

