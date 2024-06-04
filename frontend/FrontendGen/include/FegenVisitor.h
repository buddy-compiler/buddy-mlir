#ifndef FEGEN_FEGENVISITOR_H
#define FEGEN_FEGENVISITOR_H

#include <map>
#include <string>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include "FegenParserBaseVisitor.h"
#include "FegenManager.h"


using namespace antlr4;

namespace fegen{


class FegenVisitor : public FegenParserBaseVisitor {
private:
  FegenManager manager;
public:
  FegenVisitor() {}
  
};
}
#endif