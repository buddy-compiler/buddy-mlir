//===- StageManifestTest.cpp - Pipeline stage manifest parser unit tests --===//
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

#include "buddy/runtime/pipeline/StageManifest.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Attrs = std::unordered_map<std::string, std::string>;

Attrs makeValidStageAttrs(const std::string &startLayer = "0",
                          const std::string &endLayer = "14",
                          const std::string &totalLayers = "28",
                          const std::string &activationType = "f16",
                          const std::string &activationWidth = "1536",
                          const std::string &version = "1") {
  return {
      {"pp_stage_version", version},
      {"pp_start_layer", startLayer},
      {"pp_end_layer", endLayer},
      {"pp_total_layers", totalLayers},
      {"pp_activation_type", activationType},
      {"pp_activation_width", activationWidth},
  };
}

void expectEqual(int actual, int expected, const std::string &context) {
  if (actual != expected) {
    throw std::runtime_error(context + ": expected " +
                             std::to_string(expected) + ", got " +
                             std::to_string(actual));
  }
}

void expectEqual(const std::string &actual, const std::string &expected,
                 const std::string &context) {
  if (actual != expected) {
    throw std::runtime_error(context + ": expected '" + expected + "', got '" +
                             actual + "'");
  }
}

void expectParseFailure(const Attrs &attrs,
                        const std::vector<std::string> &expectedSubstrings,
                        const std::string &caseName) {
  try {
    (void)buddy::runtime::parseStageV1(attrs);
  } catch (const std::runtime_error &err) {
    const std::string message = err.what();
    for (const auto &needle : expectedSubstrings) {
      if (message.find(needle) == std::string::npos) {
        throw std::runtime_error(caseName + ": diagnostic '" + message +
                                 "' did not contain '" + needle + "'");
      }
    }
    return;
  }
  throw std::runtime_error(caseName + ": expected parseStageV1 to throw");
}

void testValidStages() {
  struct ValidCase {
    const char *name;
    const char *start;
    const char *end;
    const char *total;
    int expectedStart;
    int expectedEnd;
    int expectedTotal;
  };
  const std::vector<ValidCase> cases = {
      {"first-stage", "0", "7", "28", 0, 7, 28},
      {"middle-stage", "7", "14", "28", 7, 14, 28},
      {"last-stage", "21", "28", "28", 21, 28, 28},
      {"single-full-stage", "0", "28", "28", 0, 28, 28},
  };

  for (const auto &tc : cases) {
    Attrs attrs = makeValidStageAttrs(tc.start, tc.end, tc.total);
    // Unrelated module attributes (such as model_name, vocab_uri,
    // runner_library) must be ignored by parseStageV1.
    attrs["model_name"] = "deepseek_r1_f16";
    attrs["runner_library"] = "file:deepseek_r1_pipeline_runner.so";
    if (tc.expectedStart == 0)
      attrs["vocab_uri"] = "file:vocab.txt";

    const auto parsed = buddy::runtime::parseStageV1(attrs);
    expectEqual(parsed.startLayer, tc.expectedStart,
                std::string(tc.name) + ".startLayer");
    expectEqual(parsed.endLayer, tc.expectedEnd,
                std::string(tc.name) + ".endLayer");
    expectEqual(parsed.totalLayers, tc.expectedTotal,
                std::string(tc.name) + ".totalLayers");
    expectEqual(parsed.activationType, "f16",
                std::string(tc.name) + ".activationType");
    expectEqual(parsed.activationWidth, 1536,
                std::string(tc.name) + ".activationWidth");
  }
}

void testMissingKeys() {
  const std::vector<std::string> requiredKeys = {
      "pp_stage_version", "pp_start_layer",     "pp_end_layer",
      "pp_total_layers",  "pp_activation_type", "pp_activation_width",
  };
  for (const auto &key : requiredKeys) {
    Attrs attrs = makeValidStageAttrs();
    attrs.erase(key);
    expectParseFailure(attrs, {"missing stage field", key}, "missing-" + key);
  }

  // Whole-model manifest attributes without any pp_* keys must be rejected.
  Attrs wholeModelAttrs = {
      {"version", "0.1.0"},
      {"model_name", "deepseek_r1_f16"},
      {"vocab_uri", "file:vocab.txt"},
  };
  expectParseFailure(wholeModelAttrs,
                     {"missing stage field", "pp_stage_version"},
                     "whole-model-manifest");
}

void testInvalidVersionAndDtype() {
  for (const std::string &badVersion : {"", "0", "2", "v1", "1 "}) {
    Attrs attrs =
        makeValidStageAttrs("0", "14", "28", "f16", "1536", badVersion);
    expectParseFailure(attrs, {"pp_stage_version"},
                       "bad-version-" + badVersion);
  }

  for (const std::string &badDtype : {"", "f32", "bf16", "FP16", "int8"}) {
    Attrs attrs = makeValidStageAttrs("0", "14", "28", badDtype);
    expectParseFailure(attrs, {"pp_activation_type"}, "bad-dtype-" + badDtype);
  }
}

void testInvalidIntervalsAndBounds() {
  // Negative start_layer.
  expectParseFailure(makeValidStageAttrs("-1", "14", "28"), {"pp_start_layer"},
                     "negative-start-layer");

  // Empty interval (end == start) and inverted interval (end < start).
  expectParseFailure(makeValidStageAttrs("14", "14", "28"),
                     {"pp_end_layer", "pp_start_layer"}, "empty-interval");
  expectParseFailure(makeValidStageAttrs("14", "7", "28"),
                     {"pp_end_layer", "pp_start_layer"}, "inverted-interval");

  // total_layers < end_layer.
  expectParseFailure(makeValidStageAttrs("0", "28", "14"),
                     {"pp_total_layers", "pp_end_layer"}, "total-lt-end");

  // Non-positive activation_width.
  expectParseFailure(makeValidStageAttrs("0", "14", "28", "f16", "0"),
                     {"pp_activation_width"}, "zero-width");
  expectParseFailure(makeValidStageAttrs("0", "14", "28", "f16", "-1536"),
                     {"pp_activation_width"}, "negative-width");
}

void testMalformedIntegers() {
  const std::vector<std::string> intFields = {"pp_start_layer", "pp_end_layer",
                                              "pp_total_layers",
                                              "pp_activation_width"};
  const std::vector<std::string> badValues = {
      "", "abc", "7abc", " 7", "7 ", "99999999999999999999"};

  for (const auto &field : intFields) {
    for (const auto &bad : badValues) {
      Attrs attrs = makeValidStageAttrs();
      attrs[field] = bad;
      expectParseFailure(attrs, {field, "integer"},
                         "malformed-" + field + "-" + bad);
    }
  }
}

} // namespace

int main() {
  try {
    testValidStages();
    testMissingKeys();
    testInvalidVersionAndDtype();
    testInvalidIntervalsAndBounds();
    testMalformedIntegers();
  } catch (const std::exception &err) {
    std::cerr << "StageManifestTest FAILED: " << err.what() << "\n";
    return 1;
  }

  std::cout << "StageManifest tests passed\n";
  return 0;
}
