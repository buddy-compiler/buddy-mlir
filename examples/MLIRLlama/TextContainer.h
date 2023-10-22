//===- TextContainer.h ----------------------------------------------------===//
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
//
// Text container descriptor.
//
//===----------------------------------------------------------------------===//

#ifndef FRONTEND_INTERFACES_BUDDY_LLM_TEXTCONTAINER
#define FRONTEND_INTERFACES_BUDDY_LLM_TEXTCONTAINER

#include "buddy/Core/Container.h"
#include <fstream>
#include <iostream>
#include <unordered_map>

namespace buddy {

// Text container.
// - T represents the type of the elements.
// - N represents the number of dimensions.
template <typename T, size_t N> class Text : public MemRef<T, N> {
public:
  // Text Constructor with string.
  // This constructor initializes a Text object with the provided string.
  // The provided string is stored internally for tokenization and processing.
  Text(const std::string &str);
  // Tokenizer
  // Tokenize an input string based on vocabulary and container size.
  // This function initializes the necessary memory references and sets up the
  // structure for storing tokenized data.
  // The vocabulary file is read to build a token-to-id map for efficient token
  // processing.
  // The input string is iterated character by character, and tokens are
  // extracted based on whitespace and punctuation.
  // Tokens are processed using the `processToken` function and stored in the
  // allocated memory.
  // Special tokens (e.g., [CLS] and [SEP]) are added at the beginning and end
  // of the tokenized sequence.
  void tokenize(const std::string &vocab, long long length);
  // Get sequence length
  long long getTokenCnt();
  // Set sequence length
  void setTokenCnt(long long cnt);

private:
  // Check if a character is a whitespace character.
  bool isWhitespace(char s) const {
    return s == ' ' || s == '\t' || s == '\n' || s == '\r';
  }
  // Check if a character is a punctuation character.
  bool isPunctuation(char s) const {
    return (s >= 33 && s <= 47) || (s >= 58 && s <= 64) ||
           (s >= 91 && s <= 96) || (s >= 123 && s <= 126);
  }
  // Process a token and store its corresponding value in the container.
  // This function takes a token as input and find its corresponding value in
  // the token-to-id map.
  // If the token exists in the map, the corresponding value is stored in the
  // container at the current token count index.
  // If the token is not found in the map, the value 100 (corresponding to the
  // unknown token [UKN]) is stored in the container.
  // Finally, the token count is incremented.
  void processToken(const std::string &token, long long &tokenCnt);
  // The string member of the text container.
  std::string str;
  // Token-ID map holds the given vocabulary.
  // Since the map is only used for quick lookups and not iterating through it
  // in a specific order, using `std::unordered_map` for faster lookups.
  std::unordered_map<std::string, long long> tokenToIdMap;
  // the real length of sequence
  long long tokencnt;
};

// Text Constructor with string.
template <typename T, size_t N>
Text<T, N>::Text(const std::string &str) : MemRef<T, N>(), str(str) {}

// Tokenizer
template <typename T, size_t N>
void Text<T, N>::tokenize(const std::string &vocab, long long length) {
  // Initialize MemRef container members.
  this->offset = 0;
  this->sizes[0] = 1;
  this->sizes[1] = length;
  this->setStrides();
  this->size = this->product(this->sizes);
  this->allocated = new T[this->size];
  this->aligned = this->allocated;
  // Register vocabulary.
  long long index = 0;
  std::string value;
  std::ifstream fin(vocab);
  while (getline(fin, value)) {
    this->tokenToIdMap[value] = index++;
  }
  fin.close();
  // Tokenize string and convert to MemRef container object.
  this->aligned[0] = 1; // [CLS] NLP Classification Marker
  long long tokenCnt = 1;
  std::string token;
  for (char s : str) {
    if (isWhitespace(s) || isPunctuation(s)) {
      if (!token.empty()) {
        token = "▁" + token;
        std::cout<<token<<std::endl;
        processToken(token, tokenCnt);
        token.clear();
      }
      if (isPunctuation(s)) {
        token = s;
        std::cout<<token<<std::endl;
        processToken(token, tokenCnt);
        token.clear();
      }
    } else {
      token += s;
    }
  }
  // Parse the last token if exists.
  if (!token.empty()) {
    processToken(token, tokenCnt);
    token.clear();
  }
  // [SEP] NLP Separator Marker
  for (long long i = tokenCnt; i < length; i++) {
    this->aligned[i] = 2;
  }
  this->tokencnt = tokenCnt;
}

template <typename T, size_t N>
void Text<T, N>::processToken(const std::string &token, long long &tokenCnt) {
  if (tokenToIdMap.count(token) > 0) {
    // Stores the value corresponding to the token string into the container.
    this->aligned[tokenCnt] = tokenToIdMap[token];
  } else {
    // The token is not included in the vocabulary.
    // Assign 100 ([UKN]) to the text container.
    this->aligned[tokenCnt] = 0;
  }
  tokenCnt++;
}

template <typename T, size_t N>
long long Text<T, N>::getTokenCnt(){
  return this->tokencnt;
}

template <typename T, size_t N>
void Text<T, N>::setTokenCnt(long long cnt){
  this->tokencnt = cnt;
}

} // namespace buddy

#endif // FRONTEND_INTERFACES_BUDDY_LLM_TEXTCONTAINER
