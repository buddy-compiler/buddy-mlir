//===- Reporter.cpp - Benchmark result output for buddy-bench ------------===//
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

#include "Reporter.h"
#include "BenchBackend.h"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace buddy {
namespace bench {

namespace {

/// JSON number; non-finite becomes null (JSON has no NaN / Infinity).
std::string jsonNum(double v) {
  if (!std::isfinite(v))
    return "null";
  std::ostringstream os;
  os << std::setprecision(6) << v;
  return os.str();
}

/// Format a double for the human-readable md / csv tables.
std::string textNum(double v) {
  std::ostringstream os;
  os << std::setprecision(6) << v;
  return os.str();
}

/// Escape a string for a JSON string literal, including control characters.
std::string jsonEscape(const std::string &s) {
  static const char kHex[] = "0123456789abcdef";
  std::string out;
  for (unsigned char c : s) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\b':
      out += "\\b";
      break;
    case '\f':
      out += "\\f";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (c < 0x20) {
        out += "\\u00";
        out += kHex[c >> 4];
        out += kHex[c & 0x0f];
      } else {
        out += static_cast<char>(c);
      }
      break;
    }
  }
  return out;
}

/// Quote a CSV field when it contains a delimiter, quote or line break.
std::string csvField(const std::string &s) {
  if (s.find_first_of(",\"\n\r") == std::string::npos)
    return s;
  std::string out = "\"";
  for (char c : s) {
    if (c == '"')
      out += "\"\"";
    else
      out += c;
  }
  out += '"';
  return out;
}

/// Escape a Markdown table cell: a literal '|' would split the row.
std::string mdField(const std::string &s) {
  std::string out;
  for (char c : s) {
    if (c == '|')
      out += "\\|";
    else if (c == '\n' || c == '\r')
      out += ' ';
    else
      out += c;
  }
  return out;
}

/// Extra metrics as "name=avg+/-sd; ..." for the md / csv columns.
std::string extraMetricsText(const BenchmarkRecord &r) {
  std::ostringstream os;
  for (size_t i = 0; i < r.extraMetrics.size(); ++i) {
    if (i)
      os << "; ";
    os << r.extraMetrics[i].name << '=' << textNum(r.extraMetrics[i].avg)
       << "+/-" << textNum(r.extraMetrics[i].stddev);
  }
  return os.str();
}

/// Compact JSON object with the host / model configuration.
std::string envJson(const EnvInfo &env) {
  std::ostringstream os;
  os << "{\"cpu_model\":\"" << jsonEscape(env.cpuModel) << "\","
     << "\"cores\":" << env.numCores << ","
     << "\"used_cores\":" << env.usedCores << ","
     << "\"cpu_affinity\":\"" << jsonEscape(env.cpuAffinity) << "\","
     << "\"numa\":\"" << jsonEscape(env.numaConfig) << "\","
     << "\"isa\":\"" << jsonEscape(env.isa) << "\","
     << "\"version\":\"" << jsonEscape(env.version) << "\","
     << "\"model_path\":\"" << jsonEscape(env.modelPath) << "\","
     << "\"model_size\":" << env.modelSizeBytes << ","
     << "\"model_mtime\":\"" << jsonEscape(env.modelMtime) << "\","
     << "\"precision\":\"" << jsonEscape(env.precision) << "\","
     << "\"backend\":\"" << jsonEscape(env.backend) << "\","
     << "\"device\":\"" << jsonEscape(env.device) << "\","
     << "\"device_arch\":\"" << jsonEscape(env.deviceArch) << "\","
     << "\"device_count\":" << env.deviceCount << ","
     << "\"batch_size\":" << env.batchSize << "}";
  return os.str();
}

/// Compact JSON object for extra metrics: {"name":{"avg":..,"stddev":..},...}.
std::string extraMetricsJson(const BenchmarkRecord &r) {
  std::ostringstream os;
  os << '{';
  for (size_t i = 0; i < r.extraMetrics.size(); ++i) {
    if (i)
      os << ',';
    os << '"' << jsonEscape(r.extraMetrics[i].name)
       << "\":{\"avg\":" << jsonNum(r.extraMetrics[i].avg)
       << ",\"stddev\":" << jsonNum(r.extraMetrics[i].stddev) << '}';
  }
  os << '}';
  return os.str();
}

/// Compact JSON object for one measurement row (shared by json/jsonl).
std::string recordJson(const BenchmarkRecord &r) {
  std::ostringstream os;
  os << "{\"model\":\"" << jsonEscape(r.model) << "\","
     << "\"test\":\"" << jsonEscape(r.test) << "\","
     << "\"n_prompt\":" << r.nPrompt << ","
     << "\"n_gen\":" << r.nGen << ","
     << "\"prefill_tps\":" << jsonNum(r.prefillTps) << ","
     << "\"decode_tps\":" << jsonNum(r.decodeTps) << ","
     << "\"prefill_tps_sd\":" << jsonNum(r.prefillTpsStddev) << ","
     << "\"decode_tps_sd\":" << jsonNum(r.decodeTpsStddev) << ","
     << "\"avg_s\":" << jsonNum(r.avgSecs) << ","
     << "\"stddev_s\":" << jsonNum(r.stddevSecs) << ","
     << "\"reps\":" << r.reps << ","
     << "\"extra_metrics\":" << extraMetricsJson(r) << "}";
  return os.str();
}

} // namespace

std::string renderMarkdown(const std::vector<BenchmarkRecord> &records) {
  std::ostringstream os;
  os << "| model | test | n_prompt | n_gen | prefill_tps | decode_tps | "
        "prefill_tps_sd | decode_tps_sd | avg_s | stddev_s | reps | "
        "extra_metrics |\n";
  os << "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- |\n";
  for (const auto &r : records) {
    os << "| " << mdField(r.model) << " | " << mdField(r.test) << " | "
       << r.nPrompt << " | " << r.nGen << " | " << textNum(r.prefillTps)
       << " | " << textNum(r.decodeTps) << " | " << textNum(r.prefillTpsStddev)
       << " | " << textNum(r.decodeTpsStddev) << " | " << textNum(r.avgSecs)
       << " | " << textNum(r.stddevSecs) << " | " << r.reps << " | "
       << mdField(extraMetricsText(r)) << " |\n";
  }
  return os.str();
}

std::string renderCsv(const std::vector<BenchmarkRecord> &records) {
  std::ostringstream os;
  os << "model,test,n_prompt,n_gen,prefill_tps,decode_tps,prefill_tps_sd,"
        "decode_tps_sd,avg_s,stddev_s,reps,extra_metrics"
        "\n";
  for (const auto &r : records) {
    os << csvField(r.model) << ',' << csvField(r.test) << ',' << r.nPrompt
       << ',' << r.nGen << ',' << textNum(r.prefillTps) << ','
       << textNum(r.decodeTps) << ',' << textNum(r.prefillTpsStddev) << ','
       << textNum(r.decodeTpsStddev) << ',' << textNum(r.avgSecs) << ','
       << textNum(r.stddevSecs) << ',' << r.reps << ','
       << csvField(extraMetricsText(r)) << '\n';
  }
  return os.str();
}

std::string renderJson(const std::vector<BenchmarkRecord> &records,
                       const EnvInfo &env) {
  std::ostringstream os;
  os << "{\n";
  os << "  \"env\": " << envJson(env) << ",\n";
  os << "  \"results\": [\n";
  for (size_t i = 0; i < records.size(); ++i) {
    os << "    " << recordJson(records[i])
       << (i + 1 < records.size() ? "," : "") << "\n";
  }
  os << "  ]\n";
  os << "}\n";
  return os.str();
}

std::string renderJsonl(const std::vector<BenchmarkRecord> &records,
                        const EnvInfo &env) {
  std::ostringstream os;
  // One leading env line, then one self-contained line per measurement.
  os << "{\"env\":" << envJson(env) << "}\n";
  for (const auto &r : records)
    os << recordJson(r) << "\n";
  return os.str();
}

} // namespace bench
} // namespace buddy
