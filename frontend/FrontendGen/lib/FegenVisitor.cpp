#include "FegenVisitor.h"

bool fegen::checkParams(std::vector<fegen::FegenValue *> &expected,
                        std::vector<fegen::FegenValue *> &actual) {
  return true;
}

bool fegen::checkListLiteral(
    std::vector<std::shared_ptr<fegen::FegenRightValue::Expression>>
        &listLiteral) {
  return true;
}