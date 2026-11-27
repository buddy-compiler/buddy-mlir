// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef BUDDY_EXAMPLES_TRITONSOFTMAX_RISCVCAPABILITIES_H
#define BUDDY_EXAMPLES_TRITONSOFTMAX_RISCVCAPABILITIES_H

#include <cctype>
#include <istream>
#include <set>
#include <sstream>
#include <string>

namespace buddy::examples::softmax {

// Parse extension identities, never substrings: Zve/Zv subsets do not imply V.
// cpuinfo may use either rv64imafdcv or rv64i2p1_m2p0_..._v1p0.
inline std::set<std::string> isaExtensions(std::string isa) {
  std::set<std::string> extensions;
  if (isa.rfind("rv64", 0) != 0)
    return extensions;
  isa.erase(0, 4);
  std::istringstream pieces(isa);
  std::string piece;
  bool first = true;
  while (std::getline(pieces, piece, '_')) {
    if (first) {
      size_t i = 0;
      while (i < piece.size()) {
        if (piece[i] < 'a' || piece[i] > 'z')
          return {};
        std::string extension(1, piece[i++]);
        extensions.insert(extension);
        const size_t versionBegin = i;
        while (i < piece.size() &&
               (std::isdigit(static_cast<unsigned char>(piece[i])) ||
                piece[i] == 'p'))
          ++i;
        if (extension == "v" && i != versionBegin &&
            piece.substr(versionBegin, i - versionBegin) != "1p0")
          return {};
      }
      first = false;
    } else {
      auto version = piece.find_first_of("0123456789");
      const auto extension = piece.substr(0, version);
      if (extension == "v" && version != std::string::npos &&
          piece.substr(version) != "1p0")
        return {};
      extensions.insert(extension);
    }
  }
  if (extensions.count("g"))
    for (const char *extension : {"i", "m", "a", "f", "d", "zicsr", "zifencei"})
      extensions.insert(extension);
  return extensions;
}

// Conservative fallback capability source: every reported hart must satisfy
// every requested extension. Kernel/OS vector enablement is checked separately.
inline bool allHartsSupport(std::istream &cpuinfo,
                            const std::string &required) {
  std::istringstream requirementStream(required);
  std::set<std::string> requirements;
  std::string token;
  while (std::getline(requirementStream, token, ','))
    if (!token.empty())
      requirements.insert(token);
  if (requirements.empty())
    return false;
  unsigned harts = 0, processors = 0;
  std::string line;
  while (std::getline(cpuinfo, line)) {
    auto colon = line.find(':');
    if (colon == std::string::npos)
      continue;
    auto key = line.substr(0, colon);
    const auto begin = key.find_first_not_of(" \t");
    if (begin == std::string::npos)
      continue;
    key = key.substr(begin, key.find_last_not_of(" \t") - begin + 1);
    if (key == "processor")
      ++processors;
    if (key != "isa")
      continue;
    std::istringstream value(line.substr(colon + 1));
    std::string isa;
    value >> isa;
    const auto extensions = isaExtensions(isa);
    for (const auto &extension : requirements)
      if (!extensions.count(extension))
        return false;
    ++harts;
  }
  return harts > 0 && harts == processors;
}
} // namespace buddy::examples::softmax
#endif
