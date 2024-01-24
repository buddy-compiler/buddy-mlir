#ifndef FEGEN_IR_H
#define FEGEN_IR_H

#include <string>

namespace fegen {
enum class IRKind { TYPE, ATTRIBUTE, OPERATION };
struct FegenIR {
  IRKind kind;
  std::string name;
};
} // namespace fegen

#endif