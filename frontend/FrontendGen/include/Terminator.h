//====- Terminator.h -------------------------------------------------------===//
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

#ifndef INCLUDE_TERMINATOR_H
#define INCLUDE_TERMINATOR_H
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace frontendgen {
class CGModule;
/// A class store antlr's terminators.
class Terminators {
  friend class CGModule;

private:
  llvm::StringMap<llvm::StringRef> terminators;
  llvm::SmallSet<llvm::StringRef, 10> customTerminators;

public:
  Terminators() {
#define terminator(NAME, VALUE) terminators.insert(std::pair(#NAME, #VALUE));
#include "Terminator.def"
  }
  /// Determine if it is a terminator.
  bool isTerminator(llvm::StringRef terminator) {
    std::string tmp = terminator.str();
    tmp[0] += 32;
    if (customTerminators.contains(tmp))
      return true;
    if (terminators.find(terminator) == terminators.end())
      return false;
    return true;
  }

  void addCustomTerminators(llvm::StringRef terminator) {
    customTerminators.insert(terminator);
  }
  void addTerminator(llvm::StringRef terminator) {
    terminators.insert(std::pair(terminator, terminator));
  }
  /// Output all terminators.
  void lookTerminators() {
    llvm::outs() << "customTerminators\n";
    for (llvm::StringRef terminator : customTerminators) {
      std::string terminatorName = terminator.str();
      terminatorName[0] -= 32;
      llvm::outs() << "terminator name:" << terminatorName << ' '
                   << "terminator content:" << terminator << '\n';
    }
    for (auto start = terminators.begin(); start != terminators.end();
         ++start) {
      llvm::outs() << "terminator name:" << start->first() << ' '
                   << "terminator content:" << start->second << '\n';
    }
  }
};

} // namespace frontendgen

#endif
