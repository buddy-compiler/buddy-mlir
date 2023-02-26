//====- Lexer.cpp  --------------------------------------------------------===//
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

#include "Lexer.h"
#include "llvm/Support/raw_ostream.h"
using namespace frontendgen;
/// some function about handing characters.
namespace charinfo {
inline bool isASCLL(char ch) { return static_cast<unsigned char>(ch) <= 127; }

inline bool isWhitespace(char ch) {
  return isASCLL(ch) && (ch == ' ' || ch == '\t' || ch == '\f' || ch == '\v' ||
                         ch == '\r' || ch == '\n');
}

inline bool isIdentifierHead(char ch) {
  return isASCLL(ch) &&
         (ch == '_' || (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z'));
}

inline bool isDigit(char ch) { return isASCLL(ch) && (ch >= '0' && ch <= '9'); }

inline bool isIdentifierBody(char ch) {
  return isIdentifierHead(ch) || isDigit(ch);
}
} // namespace charinfo

/// Add keyword to keywordmap.
void KeyWordManager::addKeyWord(llvm::StringRef name, tokenKinds kind) {
  keywordMap.insert(std::make_pair(name, kind));
}

/// A function add all keywords.
void KeyWordManager::addKeyWords() {
#define KEYWORD(NAME, FLAG) addKeyWord(#NAME, tokenKinds::kw_##NAME);
#include "Token.def"
}

/// Determine if a string is a keyword.
tokenKinds KeyWordManager::getKeyWord(llvm::StringRef name, tokenKinds kind) {
  auto result = keywordMap.find(name);
  if (result != keywordMap.end())
    return result->second;
  return kind;
}

bool Token::is(tokenKinds kind) { return kind == tokenKind; }

llvm::SMLoc Token::getLocation() { return llvm::SMLoc::getFromPointer(start); }
//// Get next token.
void Lexer::next(Token &token) {
  // Skip whitespace.
  while (*curPtr && charinfo::isWhitespace(*curPtr))
    curPtr++;
  if (!*curPtr) {
    token.setTokenKind(tokenKinds::eof);
    return;
  }
  // Get identifier.
  if (charinfo::isIdentifierHead(*curPtr)) {
    identifier(token);
    return;
  } else if (charinfo::isDigit(*curPtr)) {
    number(token);
    return;
  } else if (*curPtr == ';') {
    formToken(token, curPtr + 1, tokenKinds::semi);
    return;
  } else if (*curPtr == ':') {
    formToken(token, curPtr + 1, tokenKinds::colon);
    return;
  } else if (*curPtr == '\'') {
    formToken(token, curPtr + 1, tokenKinds::apostrophe);
    return;
  } else if (*curPtr == '(') {
    formToken(token, curPtr + 1, tokenKinds::parentheseOpen);
    return;
  } else if (*curPtr == ')') {
    formToken(token, curPtr + 1, tokenKinds::parentheseClose);
    return;
  } else if (*curPtr == '*') {
    formToken(token, curPtr + 1, tokenKinds::asterisk);
    return;
  } else if (*curPtr == '?') {
    formToken(token, curPtr + 1, tokenKinds::questionMark);
    return;
  } else if (*curPtr == '+') {
    formToken(token, curPtr + 1, tokenKinds::plus);
    return;
  } else if (*curPtr == '=') {
    formToken(token, curPtr + 1, tokenKinds::equal);
    return;
  } else if (*curPtr == '{') {
    formToken(token, curPtr + 1, tokenKinds::curlyBlacketOpen);
    return;
  } else if (*curPtr == '}') {
    formToken(token, curPtr + 1, tokenKinds::curlyBlacketClose);
    return;
  } else if (*curPtr == '$') {
    formToken(token, curPtr + 1, tokenKinds::dollar);
    return;
  } else if (*curPtr == ',') {
    formToken(token, curPtr + 1, tokenKinds::comma);
    return;
  } else if (*curPtr == '<') {
    formToken(token, curPtr + 1, tokenKinds::angleBracketOpen);
    return;
  } else if (*curPtr == '>') {
    formToken(token, curPtr + 1, tokenKinds::angleBracketClose);
    return;
  } else if (*curPtr == '[') {
    formToken(token, curPtr + 1, tokenKinds::squareBracketOpen);
    return;
  } else if (*curPtr == ']') {
    formToken(token, curPtr + 1, tokenKinds::squareBracketClose);
    return;
  } else if (*curPtr == '"') {
    formToken(token, curPtr + 1, tokenKinds::doubleQuotationMark);
    return;
  }
  token.tokenKind = tokenKinds::unknown;
}

void Lexer::identifier(Token &token) {
  const char *start = curPtr;
  const char *end = curPtr + 1;
  while (charinfo::isIdentifierBody(*end))
    ++end;
  llvm::StringRef name(start, end - start);
  tokenKinds kind = keywordManager.getKeyWord(name, tokenKinds::identifier);
  formToken(token, end, kind);
}

void Lexer::formToken(Token &token, const char *tokenEnd, tokenKinds kind) {
  int length = tokenEnd - curPtr;
  token.start = curPtr;
  token.length = length;
  token.tokenKind = kind;
  curPtr = tokenEnd;
}

void Lexer::number(Token &token) {
  const char *end = curPtr;
  end++;
  while (charinfo::isDigit(*end))
    end++;
  formToken(token, end, tokenKinds::number);
}
/// Get the corresponding content according to start and end.
llvm::StringRef Lexer::getMarkContent(std::string start, std::string end) {
  while (*curPtr && charinfo::isWhitespace(*curPtr))
    curPtr++;
  int index = start.find(*curPtr);
  if (index == -1)
    return llvm::StringRef();
  char s = start[index];
  char e = end[index];
  const char *endPtr = curPtr + 1;
  int number = 1;
  if (s == e)
    while (*endPtr != e)
      endPtr++;
  else
    while (number) {
      if (*endPtr == s)
        number++;
      if (*endPtr == e)
        number--;
      if (number)
        endPtr++;
    }
  endPtr++;
  llvm::StringRef content(curPtr, endPtr - curPtr);
  curPtr = endPtr;
  return content;
}
/// Get the corresponding content according to statr and ch.
llvm::StringRef Lexer::getEndChContent(const char *start, char ch) {
  const char *endPtr = curPtr;
  while (*endPtr != ch)
    endPtr++;
  endPtr++;
  curPtr = endPtr;
  return llvm::StringRef(start, endPtr - start);
}
