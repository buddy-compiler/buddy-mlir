//====- Diagnostics.h -----------------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef INCLUDE_DIAGNOSTIC_H
#define INCLUDE_DIAGNOSTIC_H
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"

/// When there is an error in the user's code, we can diagnose the error through
/// the class.
namespace frontendgen {
class DiagnosticEngine {
  llvm::SourceMgr &SrcMgr;
  static const char *getDiagnosticText(unsigned diagID);
  llvm::SourceMgr::DiagKind getDiagnosticKind(unsigned diagID);
  bool hasReport = false;

public:
  enum diagKind {
#define DIAG(ID, Level, Msg) ID,
#include "Diagnostics.def"
  };
  DiagnosticEngine(llvm::SourceMgr &SrcMgr) : SrcMgr(SrcMgr) {}

  template <typename... Args>
  void report(llvm::SMLoc loc, unsigned diagID, Args &&...arguments) {
    if (!hasReport) {
      std::string Msg = llvm::formatv(getDiagnosticText(diagID),
                                      std::forward<Args>(arguments)...)
                            .str();
      SrcMgr.PrintMessage(loc, getDiagnosticKind(diagID), Msg);
      hasReport = true;
    }
  }
};

} // namespace frontendgen

#endif
