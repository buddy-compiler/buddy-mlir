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
  void tokenizeBert(const std::string &vocab, long long length, bool lower = true,bool affix = false);
  // LLAMA Tokenizer
  // This function initializes the necessary memory references and sets up the
  // structure for storing tokenized data. 
  // Different from the base tokenizer,this function implements the tokenize 
  // by scoring the substring and select the best matching token.
  // And we read the string at once,replace all whitespace with a special 
  // mark — thick underline.
  void tokenizeLlama(const std::string &vocab, long long length);
  
  // Revert the ids into tokens.
  // This function initializes the convert from Text memref to the string
  // which we have processed.
  // We find the tokens using ids and replace thick underline with whitespaces.
  std::string revert(Text<long long, 2> input);

  // Get sequence length
  long long getTokenCnt() { return this->tokenCnt;}
  // Set sequence length
  void setTokenCnt(long long cnt) {this->tokenCnt = cnt;}
  // Get the token string by index
  std::string getStr(long long idx) { 
    std::string str = this->idToTokenVec[idx];
    return str;
   }

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
  // Change character from uppercase to lowercase
  char toLower(char s) const{
      if(s >= 65 && s <= 90)
          return s + 32;
      else
          return s;
  }
  // Check if a char is a chinese character
  bool isChineseChar(char s) {
      const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
      int8_t highbits = static_cast<uint8_t>(s) >> 4;
      if (lookup[highbits] == 3)
          return true;
      else
          return false;
  }
  // Replace all " " with "▁"
  std::string replaceAllSpace(const std::string &str) {
    std::string tar;
    int index = 0;
    std::string replace = "▁";
    tar.append(replace);
    while(str[index]){ 
        if(str[index] != ' '){ 
            tar.append(str,index,1);
        }
        if(str[index] == ' ' && str[index-1] != ' '){ 
            tar.append(replace);
        }
            index++;
        }
        return tar;
  } 
  // Process a token and store its corresponding value in the container.
  // This function takes a token as input and find its corresponding value in
  // the token-to-id map.
  // The option affix decides if function tokenize string by affix.
  // The longest string matching strategy is adopted for word segmentation using root affixes.
  // If the root affix method is not used, we adopt the following strategy
  // If the token exists in the map, the corresponding value is stored in the
  // container at the current token count index.
  // If the token is not found in the map, the value 100 (corresponding to the
  // unknown token [UKN]) is stored in the container.
  // Finally, the token count is incremented.
  void processToken(const std::string &token, long long &tokenCnt, bool affix = false);
  // Load vocab into class
  void loadVocab(const std::string &token);
  // [UNK] NLP Padding Marker
  int pad;
  // [UNK] NLP Unknown Marker
  int unk;
  // [CLS] NLP Classification Marker
  int cls;
  // [SEP] NLP Separator Marker
  int sep;
  // The maximum input characters we can accept in one word
  long unsigned int maxInputChars = 200;
  // The string member of the text container.
  std::string str;
  // Token-ID map holds the given vocabulary.
  // Since the map is only used for quick lookups and not iterating through it
  // in a specific order, using `std::unordered_map` for faster lookups.
  std::unordered_map<std::string, long long> tokenToIdMap;
  // ID-Token vector holds the given vocabulary.
  // It is faster to find elements by index.
  std::vector<std::string> idToTokenVec;
  // Record token count. 
  long long tokenCnt;
};

// Text Constructor with string.
template <typename T, size_t N>
Text<T, N>::Text(const std::string &str) : MemRef<T, N>(), str(str) {}

template <typename T, size_t N>
void Text<T, N>::tokenizeLlama(const std::string &vocab, long long length) {
    // Initialize MemRef container members.
    this->offset = 0;
    this->sizes[0] = 1;
    this->sizes[1] = length;
    this->setStrides();
    this->size = this->product(this->sizes);
    this->allocated = new T[this->size];
    this->aligned = this->allocated;
    this->unk = 0;
    this->cls = 1;
    this->sep = 2;
    this->pad = 2;
    // Load Vocab
    loadVocab(vocab);
    str = replaceAllSpace(str);
    std::vector<long long> res;
    std::vector<float> score;
    std::vector<long long> prev;
    int len = str.length();
    score.resize(len + 1);
    prev.resize(len + 1);
    // Forward pass
    // We set the square of string's length as our score,using dynamic 
    // programming as the main algorithm to adapt the longest charactors. 
    for (int i = 0; i < len; i++) {
        for (int sub_len = 1; sub_len <= len - i; sub_len++) {
            auto sub = str.substr(i, sub_len);
            auto token = tokenToIdMap.find(sub);
            if (token != tokenToIdMap.end()) {
                int token_score = sub.length() * sub.length();
                int local_score = score[i] + token_score;
                int next = i + sub_len;
                if (score[next] < local_score) {
                    score[next] = local_score;
                    prev[next] = (*token).second;
                }
            }
        }
    }
    // Backward pass
    int i = len;
    while (i > 0) {
        long long token_id = prev[i];
        res.push_back(token_id);
        auto token = idToTokenVec[token_id];
        i -= token.length();
    }
    // Reverse the data for correct
    std::reverse(res.begin(), res.end());
    this->aligned[0] = cls;
    tokenCnt = cls;
    for(;tokenCnt < (long long)res.size()+1;tokenCnt++)
        this->aligned[tokenCnt] = res[tokenCnt-1];
    // Parse the last token if exists.
    for(long long i = tokenCnt;i < length;i++)
        this->aligned[i] = pad;
}

// Tokenizer
template <typename T, size_t N>
void Text<T, N>::tokenizeBert(const std::string &vocab, long long length, bool lower, bool affix) {
  // Initialize MemRef container members.
  this->offset = 0;
  this->sizes[0] = 1;
  this->sizes[1] = length;
  this->setStrides();
  this->size = this->product(this->sizes);
  this->allocated = new T[this->size];
  this->aligned = this->allocated;
  this->pad = 0;
  this->unk = 100;
  this->cls = 101;
  this->sep = 102;
  loadVocab(vocab);
  // Tokenize string and convert to MemRef container object.
  // Mark the beginning of our token.
  this->aligned[0] = cls;
  tokenCnt = 1;
  std::string token;
  for (size_t i = 0;i < str.size(); i++) {
    char s = str[i];
    if(lower){
        s = toLower(s);
    }
    if (isWhitespace(s) || isPunctuation(s) || isChineseChar(s)) {
      if (!token.empty()) {
        processToken(token, tokenCnt, affix);
        token.clear();
      }
      if (isPunctuation(s)) {
        token = s;
        processToken(token, tokenCnt, false);
        token.clear();
      }
      if (isChineseChar(s)) {
          token.append(str,i,3);
          // if it doesn't divide by affix,divide the chinese words one by one.
          if(!affix) {
              processToken(token, tokenCnt, false);
              token.clear();
          }
          i += 2;
      }
    } else {
      token += s;
    }
  }
  // Parse the last token if exists.
  if (!token.empty()) {
    processToken(token, tokenCnt, affix);
    token.clear();
  }
  // Mark the end of our token.
  this->aligned[tokenCnt++] = sep;
  // Padding the rest token
  for (long long i = tokenCnt; i < length; i++) {
    this->aligned[i] = sep;
  }
}

template <typename T, size_t N>
std::string Text<T, N>::revert(Text<long long, 2> input){
    std::string dst;
    for(long unsigned int i = 0; i < this->size; i++){
        int id = input.getData()[i];
        if(id == 0 || id == 1 )
            continue;
        if(id == 2)
            break;
        std::string token = this->idToTokenVec[id];
        if(token.find("▁")!= std::string::npos) {
            dst.append(" ");
            dst.append(token,3);
        }
        else
            dst.append(token);
    }
    // if(dst[0] == ' ')
    //     dst.erase(0,1);
    return dst;
}

template <typename T, size_t N>
void Text<T, N>::loadVocab(const std::string &vocab){
    // Register vocabulary.
    long long index = 0;
    std::string value;
    //todo:Support for reading more vocab file types
    if(vocab.find(".txt") != std::string::npos) {
        std::ifstream fin(vocab);
        while (getline(fin, value)) {
            this->tokenToIdMap[value] = index++;
            this->idToTokenVec.push_back(value);
        }
        fin.close();
    }
}
template <typename T, size_t N>
void Text<T, N>::processToken(const std::string &token, long long &tokenCnt, bool affix) {
    if(affix){
        if(token.size() > maxInputChars) {
            this->aligned[tokenCnt] = unk;
            tokenCnt++;
            return;
        }
        long unsigned int start = 0;
        while (start < token.size()) {
            long unsigned int end = token.size();
            std::string cur_substr = "";
            while (start < end) {
                std::string substr;
                substr.append(token, start, end - start);
                if (start > 0)
                    substr = "##" + substr;
                if (tokenToIdMap.count(substr) > 0) {
                    cur_substr = substr;
                    break;
                }
                end -= 1;
            }
            if (cur_substr == "") {
                this->aligned[tokenCnt] = unk;
                tokenCnt++;
                return;
            }
            this->aligned[tokenCnt] = tokenToIdMap[cur_substr];
            tokenCnt++;
            start = end;
        }
    }
    else {
        if (tokenToIdMap.count(token) > 0) {
            // Stores the value corresponding to the token string into the container.
            this->aligned[tokenCnt] = tokenToIdMap[token];
        } else {
            // The token is not included in the vocabulary.
            // Assign [UNK] to the text container.
            this->aligned[tokenCnt] = unk;
        }
        tokenCnt++;
    }
}
} // namespace buddy

#endif // FRONTEND_INTERFACES_BUDDY_LLM_TEXTCONTAINER
