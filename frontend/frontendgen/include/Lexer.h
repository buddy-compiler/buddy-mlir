//====- Lexer.h ---------------------------------------------------------===//
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

#ifndef INCLUDE_LEXER_H
#define INCLUDE_LEXER_H
#include "Diagnostics.h"
#include "Token.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/SourceMgr.h"
namespace frontendgen {

/// Manage all keywords.
class KeyWordManager {
  llvm::StringMap<tokenKinds> keywordMap;
  void addKeyWords();

public:
  KeyWordManager() { addKeyWords(); }
  void addKeyWord(llvm::StringRef name, tokenKinds kind);
  tokenKinds getKeyWord(llvm::StringRef name, tokenKinds kind);
};

class Lexer {
  llvm::SourceMgr &srcMgr;
  DiagnosticEngine &diagnostic;
  const char *curPtr;
  llvm::StringRef curBuffer;
  KeyWordManager keywordManager;

public:
  Lexer(llvm::SourceMgr &srcMgr, DiagnosticEngine &diagnostic)
      : srcMgr(srcMgr), diagnostic(diagnostic) {
    curBuffer = srcMgr.getMemoryBuffer(srcMgr.getMainFileID())->getBuffer();
    curPtr = curBuffer.begin();
  }
  DiagnosticEngine &getDiagnostic() { return diagnostic; }
  void next(Token &token);
  void identifier(Token &token);
  void number(Token &token);
  void formToken(Token &token, const char *tokenEnd, tokenKinds kind);
  llvm::StringRef getMarkContent(std::string start, std::string end);
  llvm::StringRef getEndChContent(const char *start, char ch);
};

} // namespace frontendgen
#endif
