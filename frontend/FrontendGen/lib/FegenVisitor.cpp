#include "FegenVisitor.h"

bool fegen::checkParams(std::vector<fegen::Value *> &expected,
                        std::vector<fegen::RightValue> &actual) {
  return true;
}

bool fegen::checkListLiteral(
    std::vector<std::shared_ptr<fegen::RightValue::Expression>>
        &listLiteral) {
  return true;
}