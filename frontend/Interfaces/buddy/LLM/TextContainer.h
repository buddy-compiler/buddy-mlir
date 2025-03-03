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
// TODO-LOW: Add a generic tokenizer.
//
//===----------------------------------------------------------------------===//

#ifndef FRONTEND_INTERFACES_BUDDY_LLM_TEXTCONTAINER
#define FRONTEND_INTERFACES_BUDDY_LLM_TEXTCONTAINER

#include "buddy/Core/Container.h"
#include <cctype>
#include <fstream>
#include <iostream>
#include <unordered_map>

namespace buddy {

// Text container.
// - T represents the type of the elements.
// - N represents the number of dimensions.
template <typename T, size_t N> class Text : public MemRef<T, N> {
public:
  // Default Constructor.
  Text() : str(""), tokenCnt(0) {
    this->allocated = static_cast<T *>(malloc(InitialSize * sizeof(T)));
    if (!this->allocated) {
      throw std::bad_alloc();
    }
    this->aligned = this->allocated;
    this->sizes[0] = 1;
    this->sizes[1] = InitialSize;
    this->setStrides();
  };
  // Text Constructor with string.
  // This constructor initializes a Text object with the provided string.
  // The provided string is stored internally for tokenization and processing.
  Text(const std::string &str);
  // Bert Tokenizer
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
  void tokenizeBert(const std::string &vocab, size_t length, bool lower = true,
                    bool affix = false);
  // LLAMA Tokenizer
  // This function initializes the necessary memory references and sets up the
  // structure for storing tokenized data.
  // Different from the base tokenizer, this function implements the tokenize
  // by scoring the substring and select the best matching token.
  // Read the string at once, and replace all whitespace with a special
  // mark — thick underline.
  void tokenizeLlama(const std::string &vocab, size_t length);
  // Stable Diffusion Tokenizer
  // This function is designed for tokenizing input text for Stable Diffusion
  // models.
  void tokenizeStableDiffusion(const std::string &vocab, size_t length);
  // DeepSeekR1 Tokenizer
  // This function is designed for tokenizing input text for DeepSeekR1 models.
  void tokenizeDeepSeekR1(const std::string &vocab, size_t length);

  // Revert the ids into tokens.
  // This function initializes the conversion from Text memref to a string.
  // Tokens are identified by ids and thick underlines are replaced with
  // whitespaces.
  std::string revertLlama();
  std::string revertWhisper();
  std::string revertDeepSeekR1();

  // Get sequence length
  size_t getTokenCnt() { return this->tokenCnt; }
  // Set sequence length
  void setTokenCnt(size_t cnt) { this->tokenCnt = cnt; }
  // Get the token string by index
  std::string getStr(size_t idx) {
    std::string str = this->idToTokenVec[idx];
    return str;
  }
  // Append token index.
  void appendTokenIdx(size_t idx) {
    if (tokenCnt >= this->getSize()) {
      resize();
    }
    this->aligned[tokenCnt++] = idx;
  }
  // Load vocab into class
  void loadVocab(const std::string &token);

private:
  // Check if a char is component of multi-bytes string.
  // Using lookup table to determine the number of bytes of a character.
  // If the number of bytes is 1, return false(0), otherwise return the
  // number of bytes.
  int isMutiBytesChar(char s) {
    const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    int8_t highbits = static_cast<uint8_t>(s) >> 4;
    if (lookup[highbits] == 1) {
      return 0;
    } else
      return lookup[highbits];
  }
  // Replace all " " with "▁"
  std::string replaceAllSpace(const std::string &str) {
    std::string res;
    int index = 0;
    std::string replace = "▁";
    res.append(replace);
    for (char c : str) {
      if (c != ' ') {
        res.push_back(c);
      }
      if (c == ' ' && (index == 0 || str[index - 1] != ' ')) {
        res.append(replace);
      }
      index++;
    }
    return res;
  }

  static const size_t InitialSize = 10;

  // Resizes the allocated memory for the array by doubling its size.
  // This method uses realloc to expand the memory block, aiming to
  // accommodate more elements without losing the existing data.
  // TODO: improve this method.
  void resize() {
    size_t size = this->getSize() * 2;
    this->allocated =
        static_cast<T *>(realloc(this->allocated, size * sizeof(T)));
    this->aligned = this->allocated;
    this->sizes[1] = size;
    this->setStrides();
  }

  // Process a token and store its corresponding value in the container.
  // This function takes a token as input and find its corresponding value in
  // the token-to-id map.
  // The option affix decides if function tokenize string by affix.
  // The longest string matching strategy is adopted for word segmentation using
  // root affixes. If the root affix method is not used, the function adopt the
  // following strategy: If the token exists in the map, the corresponding value
  // is stored in the container at the current token count index. If the token
  // is not found in the map, the value 100 (corresponding to the unknown token
  // [UKN]) is stored in the container. Finally, the token count is incremented.
  void processToken(const std::string &token, size_t &tokenCnt,
                    bool affix = false);
  void tokenizeWithAffix(const std::string &token, size_t &tokenCnt);
  std::string findLongestSubToken(const std::string &token, size_t start);
  void assignTokenId(const std::string &token, size_t &tokenCnt);
  void assignTokenIdSD(const std::string &token, size_t &tokenCnt);
  // [UNK] NLP Padding Marker
  int pad;
  // [UNK] NLP Unknown Marker
  int unk;
  // [CLS] NLP Classification Marker
  int cls;
  // [SEP] NLP Separator Marker
  int sep;
  // [BOS] NLP Begin of Sentence Marker
  int bos;
  // [EOS] NLP End of Sentence Marker
  int eos;
  // The maximum number of input characters that can be accepted in one word.
  size_t maxInputChars = 200;
  // The string member of the text container.
  std::string str;
  // Token-ID map holds the given vocabulary.
  // Since the map is only used for quick lookups and not iterating through it
  // in a specific order, using `std::unordered_map` for faster lookups.
  std::unordered_map<std::string, size_t> tokenToIdMap;
  // ID-Token vector holds the given vocabulary.
  // It is faster to find elements by index.
  std::vector<std::string> idToTokenVec;
  // Record token count.
  size_t tokenCnt;
};

// Text Constructor with string.
template <typename T, size_t N>
Text<T, N>::Text(const std::string &str) : MemRef<T, N>(), str(str) {}

// LLaMA Tokenizer
template <typename T, size_t N>
void Text<T, N>::tokenizeLlama(const std::string &vocab, size_t length) {
  // Initialize MemRef container members.
  this->offset = 0;
  this->sizes[0] = 1;
  this->sizes[1] = length;
  this->setStrides();
  size_t size = this->product(this->sizes);
  this->allocated = (T *)malloc(sizeof(T) * size);
  this->aligned = this->allocated;
  this->unk = 0;
  this->cls = 1;
  this->sep = 2;
  this->pad = 2;
  // Load Vocab
  loadVocab(vocab);
  str = replaceAllSpace(str);

  int len = str.length();
  std::vector<size_t> res;
  std::vector<float> score(len + 1, 0);
  std::vector<size_t> prev(len + 1, 0);
  // Reserve space for the results.
  res.reserve(len);

  // Forward pass
  // Use dynamic programming as the main algorithm to adapt the longest
  // charactors.
  for (int i = 0; i < len; i++) {
    for (int sub_len = 1; sub_len <= len - i; sub_len++) {
      auto iter_start = str.begin() + i;
      auto iter_end = iter_start + sub_len;
      auto token = tokenToIdMap.find(std::string(iter_start, iter_end));
      if (token != tokenToIdMap.end()) {
        int token_score = sub_len * sub_len;
        int local_score = score[i] + token_score;
        int next = i + sub_len;
        if (score[next] < local_score) {
          score[next] = local_score;
          prev[next] = token->second;
        }
      }
    }
  }
  // Backward pass
  int i = len;
  while (i > 0) {
    size_t token_id = prev[i];
    res.push_back(token_id);
    i -= idToTokenVec[token_id].length();
  }

  this->aligned[0] = cls;
  tokenCnt = 1;
  // Directly fill this->aligned in reverse order.
  for (auto it = res.rbegin(); it != res.rend(); ++it) {
    this->aligned[tokenCnt++] = *it;
  }

  for (size_t i = tokenCnt; i < length; i++) {
    this->aligned[i] = pad;
  }
}

// Bert Tokenizer
template <typename T, size_t N>
void Text<T, N>::tokenizeBert(const std::string &vocab, size_t length,
                              bool lower, bool affix) {
  // Initialize MemRef container members.
  this->offset = 0;
  this->sizes[0] = 1;
  this->sizes[1] = length;
  this->setStrides();
  size_t size = this->product(this->sizes);
  this->allocated = (T *)malloc(sizeof(T) * size);
  this->aligned = this->allocated;
  this->pad = 102;
  this->unk = 100;
  this->cls = 101;
  this->sep = 102;
  loadVocab(vocab);
  // Tokenize string and convert to MemRef container object.
  // Mark the beginning of our token.
  this->aligned[0] = cls;
  tokenCnt = 1;
  std::string token;
  for (size_t i = 0; i < str.size(); i++) {
    char s = str[i];
    if (lower) {
      s = tolower(s);
    }
    if (isspace(s) || ispunct(s) || isMutiBytesChar(s)) {
      if (!token.empty()) {
        processToken(token, tokenCnt, affix);
        token.clear();
      }
      if (ispunct(s)) {
        token = s;
        processToken(token, tokenCnt, false);
        token.clear();
      }
      if (int bytes = isMutiBytesChar(s)) {
        token.append(str, i, bytes);
        // If it doesn't divide by affix, divide the Chinese words one by one.
        if (!affix) {
          processToken(token, tokenCnt, false);
          token.clear();
        }
        i += bytes - 1;
      }
    } else {
      token += s;
    }
  }

  // Parse the last token if exists.
  if (!token.empty()) {
    processToken(token, tokenCnt, affix);
  }

  // Mark the end of token stream.
  this->aligned[tokenCnt++] = sep;
  // Padding the rest text container.
  for (size_t i = tokenCnt; i < length; i++) {
    this->aligned[i] = pad;
  }
}

// StableDiffusion Tokenizer
template <typename T, size_t N>
void Text<T, N>::tokenizeStableDiffusion(const std::string &vocab,
                                         size_t length) {
  // Initialize MemRef container members.
  this->offset = 0;
  this->sizes[0] = 1;
  this->sizes[1] = length;
  this->setStrides();
  size_t size = this->product(this->sizes);
  this->allocated = (T *)malloc(sizeof(T) * size);
  this->aligned = this->allocated;
  this->pad = 0;
  this->unk = 49407;
  this->bos = 49406;
  this->eos = 49407;
  loadVocab(vocab);
  // Tokenize string and convert to MemRef container object.
  // Mark the beginning of our token.
  this->aligned[0] = bos;
  tokenCnt = 1;
  std::string token;

  for (size_t i = 0; i < str.size(); ++i) {
    char c = tolower(str[i]);
    // Special match cases
    if (str.substr(i, 15) == "<|startoftext|>" ||
        str.substr(i, 13) == "<|endoftext|>") {
      if (!token.empty()) {
        assignTokenIdSD(token, tokenCnt);
        token.clear();
      }
      size_t len = (str.substr(i, 15) == "<|startoftext|>") ? 15 : 13;
      assignTokenIdSD(str.substr(i, len), tokenCnt);
      i += len - 1;
    }
    // Handle contractions
    else if (c == '\'' &&
             (str.substr(i, 2) == "'s" || str.substr(i, 2) == "'t" ||
              str.substr(i, 3) == "'re" || str.substr(i, 3) == "'ve" ||
              str.substr(i, 2) == "'m" || str.substr(i, 3) == "'ll" ||
              str.substr(i, 2) == "'d")) {
      if (!token.empty()) {
        assignTokenIdSD(token, tokenCnt);
        token.clear();
      }
      size_t len = (str.substr(i, 3) == "'re" || str.substr(i, 3) == "'ve" ||
                    str.substr(i, 3) == "'ll")
                       ? 3
                       : 2;
      assignTokenIdSD(str.substr(i, len), tokenCnt);
      i += len - 1;
    }
    // Handle letters
    else if (std::isalpha(static_cast<unsigned char>(c))) {
      token += c;
    }
    // Handle digits
    else if (std::isdigit(static_cast<unsigned char>(c))) {
      token += c;
    }
    // Handle other characters
    else {
      if (!token.empty()) {
        assignTokenIdSD(token, tokenCnt);
        token.clear();
      }
      token += c;
      if (token != " ")
        assignTokenIdSD(token, tokenCnt);
      token.clear();
    }
  }
  // Parse the last token if exists.
  if (!token.empty()) {
    assignTokenIdSD(token, tokenCnt);
  }

  // Mark the end of token stream.
  this->aligned[tokenCnt++] = eos;
  // Padding the rest text container.
  for (size_t i = tokenCnt; i < length; i++) {
    this->aligned[i] = pad;
  }
}

// DeepSeekR1 Tokenizer
template <typename T, size_t N>
void Text<T, N>::tokenizeDeepSeekR1(const std::string &vocab, size_t length) {
  // Initialize MemRef container members.
  this->offset = 0;
  this->sizes[0] = 1;
  this->sizes[1] = length;
  this->setStrides();
  size_t size = this->product(this->sizes);
  this->allocated = (T *)malloc(sizeof(T) * size);
  this->aligned = this->allocated;
  this->bos = 151646;
  this->eos = 151643;
  this->pad = 151643;
  const int userToken = 151644;
  const int assistantToken = 151645;
  const int thinkToken = 151648;
  
  tokenCnt = 0;
  this->aligned[tokenCnt++] = bos;
  this->aligned[tokenCnt++] = bos;
  this->aligned[tokenCnt++] = userToken;

  // Load Vocab
  loadVocab(vocab);

  // Replace space with Ġ.
  std::string strWithoutSpace;
  std::string replace = "Ġ";
  for (int i = 0; i < (int)str.size(); i++) {
    if (str[i] != ' ')
      strWithoutSpace.push_back(str[i]);
    if (str[i] == ' ' && str[i - 1] != ' ')
      strWithoutSpace.append(replace);
  }

  int len = strWithoutSpace.length();
  std::vector<size_t> res;
  std::vector<float> score(len + 1, 0);
  std::vector<size_t> prev(len + 1, 0);
  // Reserve space for the results.
  res.reserve(len);

  // Forward pass
  // Use dynamic programming as the main algorithm to adapt the longest
  // charactors.
  for (int i = 0; i < len; i++) {
    for (int sub_len = 1; sub_len <= len - i; sub_len++) {
      auto iter_start = strWithoutSpace.begin() + i;
      auto iter_end = iter_start + sub_len;
      auto token = tokenToIdMap.find(std::string(iter_start, iter_end));
      if (token != tokenToIdMap.end()) {
        int token_score = sub_len * sub_len;
        int local_score = score[i] + token_score;
        int next = i + sub_len;
        if (score[next] < local_score) {
          score[next] = local_score;
          prev[next] = token->second;
        }
      }
    }
  }
  // Backward pass
  int i = len;
  while (i > 0) {
    size_t token_id = prev[i];
    res.push_back(token_id);
    i -= idToTokenVec[token_id].length();
  }

  // Directly fill this->aligned in reverse order.
  for (auto it = res.rbegin(); it != res.rend(); ++it) {
    this->aligned[tokenCnt++] = *it;
  }

  this->aligned[tokenCnt++] = assistantToken;
  this->aligned[tokenCnt++] = thinkToken;
  this->aligned[tokenCnt++] = 198;

  for (size_t i = tokenCnt; i < length; i++) {
    this->aligned[i] = pad;
  }
}

// The revert function is used to convert the tokenized sequence back to a
// full string.
template <typename T, size_t N> std::string Text<T, N>::revertLlama() {
  std::string dst;

  const int PAD_ID = 0;
  const int CLS_ID = 1;
  const int SEP_ID = 2;

  for (size_t i = 0; i < this->tokenCnt; i++) {
    int id = this->aligned[i];
    if (id == PAD_ID || id == CLS_ID)
      continue;
    if (id == SEP_ID)
      break;
    // Replace each "▁" with a space.
    std::string token = this->idToTokenVec[id];
    size_t pos = token.find("▁");
    while (pos != std::string::npos) {
      token.replace(pos, 3, " ");
      pos = token.find("▁", pos + 1);
    }
    dst.append(token);
  }
  if (dst[0] == ' ') {
    dst.erase(0, 1);
  }
  return dst;
}

template <typename T, size_t N> std::string Text<T, N>::revertWhisper() {
  std::string dst;

  const int PAD_ID = 50257;
  const int CLS_ID = 50258;
  const int SEP_ID = 50257;
  const int TRAN_ID = 50359;
  const int NOTIMESTAMPS_ID = 50363;

  for (size_t i = 0; i < this->tokenCnt; i++) {
    int id = this->aligned[i];
    // pad / start / type timestamps / language
    if (id == PAD_ID || id == CLS_ID || id == TRAN_ID ||
        id == NOTIMESTAMPS_ID || (id >= 50259 && id <= 50357))
      continue;
    if (id == SEP_ID)
      break;
    // Replace each "▁" with a space.
    std::string token = this->idToTokenVec[id];
    size_t pos = token.find("Ġ");
    while (pos != std::string::npos) {
      token.replace(pos, 2, " ");
      pos = token.find("Ġ", pos + 1);
    }
    dst.append(token);
  }
  if (dst[0] == ' ') {
    dst.erase(0, 1);
  }
  return dst;
}

template <typename T, size_t N> std::string Text<T, N>::revertDeepSeekR1() {
  std::string dst;

  const int EOS_ID = 151643;

  for (size_t i = 0; i < this->tokenCnt; i++) {
    int id = this->aligned[i];
    if (id == EOS_ID)
      break;
    // Replace each "Ġ" with a space.
    std::string token = this->idToTokenVec[id];
    size_t pos = token.find("Ġ");
    while (pos != std::string::npos) {
      token.replace(pos, 2, " ");
      pos = token.find("Ġ", pos + 1);
    }
    // Replace each "Ċ" with \n.
    pos = token.find("Ċ");
    while (pos != std::string::npos) {
      token.replace(pos, 2, "\n");
      pos = token.find("Ċ", pos + 1);
    }

    dst.append(token);
  }
  if (dst[0] == ' ') {
    dst.erase(0, 1);
  }
  return dst;
}

template <typename T, size_t N>
void Text<T, N>::loadVocab(const std::string &vocab) {
  // TODO-LOW: If in the future, there are more vocab file types to support,
  // consider implementing a more advanced mechanism to determine
  // and process each file type.
  std::ifstream fin(vocab);
  if (!fin.is_open()) {
    throw std::runtime_error("Failed to open vocab file: " + vocab);
  }

  std::string token;
  size_t index = 0;

  while (getline(fin, token)) {
    tokenToIdMap[token] = index++;
    idToTokenVec.push_back(token);
  }
  fin.close();
}

template <typename T, size_t N>
void Text<T, N>::processToken(const std::string &token, size_t &tokenCnt,
                              bool affix) {
  if (affix) {
    tokenizeWithAffix(token, tokenCnt);
  } else {
    assignTokenId(token, tokenCnt);
  }
}

template <typename T, size_t N>
void Text<T, N>::tokenizeWithAffix(const std::string &token, size_t &tokenCnt) {
  if (token.size() > maxInputChars) {
    this->aligned[tokenCnt++] = unk;
    return;
  }
  size_t start = 0;
  while (start < token.size()) {
    std::string subToken = findLongestSubToken(token, start);
    if (subToken.empty()) {
      this->aligned[tokenCnt++] = unk;
      return;
    }
    this->aligned[tokenCnt++] = tokenToIdMap[subToken];
    start += subToken.size();
  }
}

template <typename T, size_t N>
std::string Text<T, N>::findLongestSubToken(const std::string &token,
                                            size_t start) {
  size_t end = token.size();
  while (start < end) {
    std::string substr = token.substr(start, end - start);
    if (start > 0) {
      substr = "##" + substr;
    }
    if (tokenToIdMap.count(substr)) {
      return substr;
    }
    end--;
  }
  return "";
}

template <typename T, size_t N>
void Text<T, N>::assignTokenId(const std::string &token, size_t &tokenCnt) {
  if (tokenToIdMap.count(token)) {
    this->aligned[tokenCnt++] = tokenToIdMap[token];
  } else {
    this->aligned[tokenCnt++] = unk;
  }
}

template <typename T, size_t N>
void Text<T, N>::assignTokenIdSD(const std::string &token, size_t &tokenCnt) {
  const std::string token_suffixed = token + "</w>";
  if (tokenToIdMap.count(token_suffixed)) {
    this->aligned[tokenCnt++] = tokenToIdMap[token_suffixed];
  } else {
    // The BPE encoding needs to be implemented here.
    this->aligned[tokenCnt++] = unk;
  }
}

} // namespace buddy

#endif // FRONTEND_INTERFACES_BUDDY_LLM_TEXTCONTAINER
