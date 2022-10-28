//====- Token.h --------------------------------------------------------===//
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

#ifndef INCLUDE_TOKEN
#define INCLUDE_TOKEN
#include "Lexer.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
namespace frontendgen {
enum tokenKinds {
#define TOK(ID) ID,
#include "Token.def"
  NUM_TOKENS
};
/// store token names.
static const char *tokenNameMap[] = {
#define TOK(ID) #ID,
#define KEYWORD(ID, FLAG) #ID,
#include "Token.def"
    nullptr};

class Token {
  friend class Lexer;

private:
  tokenKinds tokenKind;
  const char *start;
  int length;

public:
  void setTokenKind(tokenKinds kind) { tokenKind = kind; }
  void setLength(int len) { length = len; }

  llvm::StringRef getContent() { return llvm::StringRef(start, length); }
  tokenKinds getKind() { return tokenKind; }
  const char *getTokenName() { return tokenNameMap[tokenKind]; }
  bool is(tokenKinds kind);
  llvm::SMLoc getLocation();
};

} // namespace frontendgen
#endif
