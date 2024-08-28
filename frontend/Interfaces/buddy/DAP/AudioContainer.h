//===- AudioContainer.h ---------------------------------------------------===//
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
// The audio decoding process in this file references the `AudioFile` library,
// which is hereby acknowledged.
// For the license of the `AudioFile` library,
// please see: https://github.com/adamstark/AudioFile/blob/master/LICENSE
//
//===----------------------------------------------------------------------===//
//
// Audio container descriptor.
//
//===----------------------------------------------------------------------===//

#ifndef FRONTEND_INTERFACES_BUDDY_DAP_AUDIOCONTAINER
#define FRONTEND_INTERFACES_BUDDY_DAP_AUDIOCONTAINER

#include "buddy/Core/Container.h"
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>

namespace dap {
template <typename T, size_t N> class Audio : public MemRef<T, N> {
public:
  // Constructor to initialize the Audio MemRef object with a file name.
  Audio(std::string filename);
  // Constructor to convert MemRef object to Audio MemRef object. Member 
  // variables are initialized with default values.
  Audio(MemRef<T, N> &&memref) noexcept;

  // Retrieve the name of the audio format.
  std::string getFormatName() const {
    switch (this->audioFormat) {
    case AudioFormat::WAV:
      return "WAV";
    default:
      return "Unsupported format";
    }
  }
  // Returns the number of bits per sample.
  int getBitDepth() const { return static_cast<int>(this->bitsPerSample); }
  // Returns the number of samples per channel.
  size_t getSamplesNum() const { return this->numSamples; }
  // Returns the number of audio channels.
  int getChannelsNum() const { return static_cast<int>(this->numChannels); }
  // Returns the sampling rate in samples per second.
  int getSampleRate() const { return static_cast<int>(this->sampleRate); }

  // Sets the number of bits per sample.
  void setBitDepth(int bitDepth) {
    this->bitsPerSample = static_cast<uint16_t>(bitDepth);
  }
  // Sets the number of samples per channel.
  void setSamplesNum(size_t samplesNum) { this->numSamples = samplesNum; }
  // Sets the number of audio channels.
  void setChannelsNum(int channelsNum) {
    this->numChannels = static_cast<uint16_t>(channelsNum);
  }
  // Sets the sampling rate in samples per second.
  void setSampleRate(int sampleRate) {
    this->sampleRate = static_cast<uint32_t>(sampleRate);
  }

  // Create an Audio File with file name and format.
  bool saveToFile(std::string filename, std::string format);

private:
  // Sample bit depth.
  uint16_t bitsPerSample;
  // Number of samples per channel.
  size_t numSamples;
  // Number of audio channels.
  uint16_t numChannels;
  // Samples per second (Hz).
  uint32_t sampleRate;
  // Enum to represent supported audio formats.
  enum class AudioFormat {
    ERROR, // Represents an error or unsupported format.
    WAV,   // WAV format.
  } audioFormat;
  // Enum to represent byte order of data.
  enum class Endianness { LittleEndian, BigEndian };

  // Decoders for multiple audio file formats.
  // Decode a WAV file into MemRef format.
  bool decodeWaveFile(const std::vector<uint8_t> &fileData);

  // Encoders for multiple audio file formats.
  // Encode a MemRef into WAV format.
  bool EncodeWaveFile(std::vector<uint8_t> &fileData);

  // Helper functions for decoding and data manipulation
  // Find the index of a specified chunk in the audio file.
  size_t getIndexOfChunk(const std::vector<uint8_t> &fileData,
                         const std::string &chunkHeaderID, size_t startIndex,
                         Endianness endianness = Endianness::LittleEndian);
  // Convert four bytes to a 32-bit integer according to byte order of data.
  int32_t fourBytesToI32(const std::vector<uint8_t> &fileData,
                         size_t startIndex,
                         Endianness endianness = Endianness::LittleEndian);
  // Convert two bytes to a 16-bit integer according to byte order of data.
  int16_t twoBytesToI16(const std::vector<uint8_t> &fileData, size_t startIndex,
                        Endianness endianness = Endianness::LittleEndian);
  // Normalize 8-bit unsigned integer sample to a range of -1.0 to 1.0.
  T oneByteToSample(uint8_t data) {
    return static_cast<T>(data - 128) / static_cast<T>(128.);
  }
  // Normalize 16-bit signed integer sample to a range of -1.0 to 1.0.
  T twoBytesToSample(int16_t data) {
    return static_cast<T>(data) / static_cast<T>(32768.);
  }

  // Helper functions for encoding and data manipulation.
  // Converts each character in the string to a byte.
  void stringToBytes(std::vector<uint8_t> &fileData, const std::string &str) {
    for (size_t i = 0; i < str.size(); i++)
      fileData.push_back(static_cast<uint8_t>(str[i]));
  }
  // Converts a 32-bit integer to four bytes according to byte order of data.
  void i32ToFourBytes(std::vector<uint8_t> &fileData, int32_t num,
                      Endianness endianness = Endianness::LittleEndian);
  // Converts a 16-bit integer to two bytes according to byte order of data.
  void i16ToTwoBytes(std::vector<uint8_t> &fileData, int16_t num,
                     Endianness endianness = Endianness::LittleEndian);
  // Converts an audio sample to a 8-bit PCM format (one byte).
  uint8_t sampleToOneByte(T sample);
  // Converts an audio sample to a 16-bit PCM format (two bytes).
  int16_t sampleToI16(T sample);
};

// Audio Container Constructor.
// Constructs an audio container object from the audio file path.
template <typename T, std::size_t N> Audio<T, N>::Audio(std::string filePath) {
  // ---------------------------------------------------------------------------
  // 1. Read the audio file into a std::vector.
  // ---------------------------------------------------------------------------
  // Open the file in binary mode and position the file pointer at the end of
  // the file.
  std::ifstream file(filePath, std::ios::binary | std::ios::ate);
  // Check if the file was successfully opened.
  if (!file) {
    throw std::runtime_error("Error: Unable to open file at " + filePath);
  }
  // Get the size of the file.
  size_t dataLength = file.tellg();
  // Move file pointer to the beginning of the file.
  file.seekg(0, std::ios::beg);
  // Create a vector to store the data.
  std::vector<uint8_t> fileData(dataLength);
  // Read the data.
  if (!file.read(reinterpret_cast<char *>(fileData.data()), dataLength)) {
    throw std::runtime_error("Error: Unable to read data from " + filePath);
  }
  // ---------------------------------------------------------------------------
  // 2. Determine the audio format and decode the audio data into MemRef.
  // ---------------------------------------------------------------------------
  std::string header(fileData.begin(), fileData.begin() + 4);
  // Check the file header to determine the format.
  if (header == "RIFF") {
    this->audioFormat = AudioFormat::WAV;
    bool success = decodeWaveFile(fileData);
    if (!success) {
      this->audioFormat = AudioFormat::ERROR;
      throw std::runtime_error("Failed to decode WAV file from " + filePath);
    };
  } else {
    this->audioFormat = AudioFormat::ERROR;
    throw std::runtime_error("Unsupported audio format detected in file " +
                             filePath);
  }
}

// Constructs an audio container object from a MemRef object. Initializes 
// metadata with default values.
template <typename T, std::size_t N>
Audio<T, N>::Audio(MemRef<T, N> &&memref) noexcept
    : MemRef<T, N>(std::move(memref)), bitsPerSample(0), numSamples(0),
      numChannels(0), sampleRate(0) {}

// Create Audio File.
// Save Audio MemRef to the specified file path using the desired format.
template <typename T, std::size_t N>
bool Audio<T, N>::saveToFile(std::string filePath, std::string format) {
  // ---------------------------------------------------------------------------
  // 1. Determine the audio format and encode the MemRef into file data.
  // ---------------------------------------------------------------------------
  // Convert the string to lowercase before comparison, ensuring that case
  // variations are handled without repeating conditions.
  std::transform(format.begin(), format.end(), format.begin(), ::tolower);
  // Vector for storing bytes in a specific audio format.
  std::vector<uint8_t> fileData;
  // Select encoder.
  if (format == "wav" || format == "wave") {
    bool success = EncodeWaveFile(fileData);
    if (!success) {
      std::cerr << "Failed to encode WAVE file." << std::endl;
      return false;
    }
  } else {
    std::cerr << "Unsupported: The encoding method for " << format
              << " format is not yet supported." << std::endl;
    return false;
  }
  // ---------------------------------------------------------------------------
  // 2. Write std::vector into audio file.
  // ---------------------------------------------------------------------------
  std::ofstream outputFile(filePath, std::ios::binary);

  if (outputFile.is_open()) {
    for (size_t i = 0; i < fileData.size(); i++) {
      char value = static_cast<char>(fileData[i]);
      outputFile.write(&value, sizeof(char));
    }

    outputFile.close();

    return true;
  }

  return false;
}

// WAV Audio File Decoder
template <typename T, std::size_t N>
bool Audio<T, N>::decodeWaveFile(const std::vector<uint8_t> &fileData) {
  // This container class only cares about the data and key information in the
  // audio file, so only the format and data chunk are decoded here.
  // Find the starting indices of critical chunks within the WAV file.
  size_t indexOfFormatChunk = getIndexOfChunk(fileData, "fmt ", 12);
  size_t indexOfDataChunk = getIndexOfChunk(fileData, "data", 12);

  // Decode the 'format' chunk to obtain format specifications.
  // Format sub-chunk:
  //   sub-chunk ID: char[4] | 4 bytes | "fmt "
  //   sub-chunk size: uint32_t | 4 bytes
  //   audio format: uint16_t | 2 bytes | 1 for PCM
  //   number of channels: uint16_t | 2 bytes
  //   sample rate: uint32_t | 4 bytes
  //   byte rate: uint32_t | 4 bytes
  //   block align: uint16_t | 2 bytes
  //   bits per sample: uint16_t | 2 bytes
  std::string formatChunkID(fileData.begin() + indexOfFormatChunk,
                            fileData.begin() + indexOfFormatChunk + 4);
  // uint32_t fmtChunkSize = fourBytesToI32(fileData, indexOfFormatChunk + 4);
  // uint16_t audioFormat = twoBytesToI16(fileData, indexOfFormatChunk + 8);
  this->numChannels = twoBytesToI16(fileData, indexOfFormatChunk + 10);
  this->sampleRate = fourBytesToI32(fileData, indexOfFormatChunk + 12);
  // byteRate = sampleRate * numChannels * bitsPerSample / 8
  // uint32_t byteRate = fourBytesToI32(fileData, indexOfFormatChunk + 16);
  // blockAlign = numChannels * bitsPerSample / 8
  uint16_t blockAlign = twoBytesToI16(fileData, indexOfFormatChunk + 20);
  this->bitsPerSample = twoBytesToI16(fileData, indexOfFormatChunk + 22);
  uint16_t numBytesPerSample = static_cast<uint16_t>(this->bitsPerSample) / 8;

  // Decode `data` chunk.
  // Data sub-chunk:
  //   sub-chunk ID: char[4] | 4 bytes | "data"
  //   sub-chunk size: uint32_t | 4 bytes
  //   data | remains
  std::string dataChunkID(fileData.begin() + indexOfDataChunk,
                          fileData.begin() + indexOfDataChunk + 4);
  int32_t dataChunkSize = fourBytesToI32(fileData, indexOfDataChunk + 4);
  this->numSamples = dataChunkSize / blockAlign;
  // size_t numSamplesPerChannels = this->numSamples / this->numChannels;
  size_t samplesStartIndex = indexOfDataChunk + 8;

  // Audio MemRef layout defaults to 1 dimension.
  // Sample values from multiple channels are stored together.
  if (N == 1) {
    this->sizes[0] = this->numSamples;
  } else if (N == this->numChannels) {
    // TODO: add conversion from 1 dimension to multi-dimension
    std::cerr << "Unsupported: The MemRef layout of multi-dimensional channels "
                 "is not yet supported."
              << std::endl;
    return false;
  } else {
    std::cerr << "Error: dimension mismatch (audio file channel: "
              << this->numChannels << " MemRef layout channel: " << N << ")"
              << std::endl;
    return false;
  }

  // Allocate memory for MemRef.
  this->setStrides();
  size_t size = this->product(this->sizes);
  this->allocated = (T *)malloc(sizeof(T) * size);
  this->aligned = this->allocated;

  // Sample data type: 8 bit
  if (this->bitsPerSample == 8) {
    size_t memrefIndex = 0;
    for (size_t i = 0; i < this->numSamples; i++) {
      for (size_t channel = 0; channel < this->numChannels; channel++) {
        size_t sampleIndex =
            samplesStartIndex + (blockAlign * i) + channel * numBytesPerSample;
        this->aligned[memrefIndex] = oneByteToSample(fileData[sampleIndex]);
        memrefIndex++;
      }
    }
  }
  // Sample data type: 16 bit
  else if (this->bitsPerSample == 16) {
    size_t memrefIndex = 0;
    for (size_t i = 0; i < this->numSamples; i++) {
      for (size_t channel = 0; channel < this->numChannels; channel++) {
        size_t sampleIndex =
            samplesStartIndex + (blockAlign * i) + channel * numBytesPerSample;
        int16_t dataTwoBytes = twoBytesToI16(fileData, sampleIndex);
        this->aligned[memrefIndex] = twoBytesToSample(dataTwoBytes);
        memrefIndex++;
      }
    }
  }
  // Other data types are not currently supported.
  else {
    std::cerr << "Unsupported audio data type." << std::endl;
    return false;
  }

  return true;
}

// WAV Audio File Encoder
template <typename T, std::size_t N>
bool Audio<T, N>::EncodeWaveFile(std::vector<uint8_t> &fileData) {
  // Encode the 'header' chunk.
  // RIFF chunk descriptor
  //   chunk ID: char[4] | 4 bytes | "RIFF"
  //   chunk size: uint32_t | 4bytes
  //   format: char[4] | 4 bytes | "WAVE"
  stringToBytes(fileData, "RIFF");
  int16_t audioFormat = this->bitsPerSample == 32 ? 0 : 1;
  // Size for 'format' sub-chunk, doesn't include metadata length.
  int32_t formatChunkSize = audioFormat == 1 ? 16 : 18;
  // Size for 'data' sub-chunk, doesn't include metadata length.
  int32_t dataChunkSize =
      this->numSamples * this->numChannels * this->bitsPerSample / 8;
  // The file size in bytes include header chunk size(4, not counting RIFF and
  // WAVE), the format chunk size(formatChunkSize and 8 bytes for metadata), the
  // data chunk size(dataChunkSize and 8 bytes for metadata).
  int32_t fileSizeInBytes = 4 + formatChunkSize + 8 + dataChunkSize + 8;
  i32ToFourBytes(fileData, fileSizeInBytes);
  stringToBytes(fileData, "WAVE");

  // Encode the 'format' chunk.
  // Format sub-chunk:
  //   sub-chunk ID: char[4] | 4 bytes | "fmt "
  //   sub-chunk size: uint32_t | 4 bytes
  //   audio format: uint16_t | 2 bytes | 1 for PCM
  //   number of channels: uint16_t | 2 bytes
  //   sample rate: uint32_t | 4 bytes
  //   byte rate: uint32_t | 4 bytes
  //   block align: uint16_t | 2 bytes
  //   bits per sample: uint16_t | 2 bytes
  stringToBytes(fileData, "fmt ");
  i32ToFourBytes(fileData, formatChunkSize);
  i16ToTwoBytes(fileData, audioFormat);
  i16ToTwoBytes(fileData, static_cast<int16_t>(this->numChannels));
  i32ToFourBytes(fileData, static_cast<int32_t>(this->sampleRate));
  int16_t numBytesPerBlock =
      static_cast<int16_t>(dataChunkSize / this->numSamples);
  int32_t numBytesPerSecond =
      static_cast<int32_t>(this->sampleRate * numBytesPerBlock);
  i32ToFourBytes(fileData, numBytesPerSecond);
  i16ToTwoBytes(fileData, numBytesPerBlock);
  i16ToTwoBytes(fileData, static_cast<int16_t>(this->bitsPerSample));

  // Encode the 'data' chunk.
  // Data sub-chunk:
  //   sub-chunk ID: char[4] | 4 bytes | "data"
  //   sub-chunk size: uint32_t | 4 bytes
  //   data | remains
  stringToBytes(fileData, "data");
  i32ToFourBytes(fileData, dataChunkSize);

  // Sample data length: 8 bit
  if (this->bitsPerSample == 8) {
    size_t memrefIndex = 0;
    for (size_t i = 0; i < this->numSamples; i++) {
      for (size_t channel = 0; channel < this->numChannels; channel++) {
        uint8_t byte = sampleToOneByte(this->aligned[memrefIndex]);
        fileData.push_back(byte);
        memrefIndex++;
      }
    }
  }
  // Sample data length: 16 bit
  else if (this->bitsPerSample == 16) {
    size_t memrefIndex = 0;
    for (size_t i = 0; i < this->numSamples; i++) {
      for (size_t channel = 0; channel < this->numChannels; channel++) {
        int16_t sampleAsInt = sampleToI16(this->aligned[memrefIndex]);
        i16ToTwoBytes(fileData, sampleAsInt);
        memrefIndex++;
      }
    }
  }
  // Other data length are not yet supported.
  else {
    std::cerr << "Unsupported audio data length: " << this->bitsPerSample
              << " bit" << std::endl;
    return false;
  }

  return true;
}

// Locates the start index of a specific chunk in a WAV file data buffer.
// Params:
//   fileData: Vector containing the raw binary data of the WAV file.
//   chunkHeaderID: The 4-byte identifier for the chunk (e.g., "fmt ", "data").
//   startIndex: Index to start the search from within the fileData.
//   endianness: Byte order used to interpret multi-byte values in the chunk
//   size.
// Returns:
//   The index of the start of the chunk or 0 if not found.
template <typename T, std::size_t N>
size_t Audio<T, N>::getIndexOfChunk(const std::vector<uint8_t> &fileData,
                                    const std::string &chunkHeaderID,
                                    size_t startIndex, Endianness endianness) {
  constexpr int dataLen = 4;
  if (chunkHeaderID.size() != dataLen) {
    assert(false && "Chunk header ID must be exactly 4 characters long");
    return -1;
  }
  size_t i = startIndex;
  while (i < fileData.size() - dataLen) {
    // Check if the current bytes match the chunk header ID
    if (memcmp(&fileData[i], chunkHeaderID.data(), dataLen) == 0) {
      return i;
    }
    // Skip to the next chunk: advance by the size of the current chunk
    // Move index to the size part of the chunk
    i += dataLen;
    // Prevent reading beyond vector size
    if (i + dataLen > fileData.size())
      break;
    // Get the size of the chunk.
    auto chunkSize = fourBytesToI32(fileData, i, endianness);
    if (chunkSize < 0) {
      assert(false && "Invalid chunk size encountered");
      return -1;
    }
    // Move to the next chunk header
    i += (dataLen + chunkSize);
  }
  // Return 0 if the chunk is not found
  return 0;
}

// Converts four bytes from the file data array to a 32-bit integer based on
// endianness. Params:
//   fileData: Vector containing the raw binary data.
//   startIndex: Index in fileData where the 4-byte sequence starts.
//   endianness: Specifies the byte order (LittleEndian or BigEndian).
// Returns:
//   The 32-bit integer converted from the byte sequence.
template <typename T, std::size_t N>
int32_t Audio<T, N>::fourBytesToI32(const std::vector<uint8_t> &fileData,
                                    size_t startIndex, Endianness endianness) {
  // Ensure the index is within the bounds to prevent out-of-range access.
  if (startIndex + 3 >= fileData.size()) {
    throw std::out_of_range("Index out of range for fourBytesToI32");
  }
  // Use uint32_t to prevent sign extension and maintain accurate binary
  // representation during bit operations.
  uint32_t result;
  if (endianness == Endianness::LittleEndian) {
    result = (static_cast<uint32_t>(fileData[startIndex + 3]) << 24) |
             (static_cast<uint32_t>(fileData[startIndex + 2]) << 16) |
             (static_cast<uint32_t>(fileData[startIndex + 1]) << 8) |
             static_cast<uint32_t>(fileData[startIndex]);
  } else {
    result = (static_cast<uint32_t>(fileData[startIndex]) << 24) |
             (static_cast<uint32_t>(fileData[startIndex + 1]) << 16) |
             (static_cast<uint32_t>(fileData[startIndex + 2]) << 8) |
             static_cast<uint32_t>(fileData[startIndex + 3]);
  }
  // Convert the unsigned result to signed int32_t to match the function's
  // return type.
  return static_cast<int32_t>(result);
}

// Converts two bytes from the file data array to a 16-bit integer based on
// endianness. Params:
//   fileData: Vector containing the raw binary data.
//   startIndex: Index in fileData where the 2-byte sequence starts.
//   endianness: Specifies the byte order (LittleEndian or BigEndian).
// Returns:
//   The 16-bit integer converted from the byte sequence.
template <typename T, std::size_t N>
int16_t Audio<T, N>::twoBytesToI16(const std::vector<uint8_t> &fileData,
                                   size_t startIndex, Endianness endianness) {
  // Ensure the index is within the bounds to prevent out-of-range access.
  if (startIndex + 1 >= fileData.size()) {
    throw std::out_of_range("Index out of range for twoBytesToI16");
  }
  // Use uint16_t to prevent sign extension and maintain accurate binary
  // representation during bit operations.
  uint16_t result;
  if (endianness == Endianness::LittleEndian) {
    result = (static_cast<uint16_t>(fileData[startIndex + 1]) << 8) |
             static_cast<uint16_t>(fileData[startIndex]);
  } else {
    result = (static_cast<uint16_t>(fileData[startIndex]) << 8) |
             static_cast<uint16_t>(fileData[startIndex + 1]);
  }
  // Convert the unsigned result to signed int16_t to match the function's
  // return type.
  return static_cast<int16_t>(result);
}

// Converts a 32-bit integer to four bytes based on endianness.
// Params:
//   fileData: Vector containing the raw binary data.
//   num: A 32-bit integer prepared for convertion.
//   endianness: Specifies the byte order (LittleEndian or BigEndian).
template <typename T, size_t N>
void Audio<T, N>::i32ToFourBytes(std::vector<uint8_t> &fileData, int32_t num,
                                 Endianness endianness) {
  // Use uint8_t to prevent sign extension and maintain accurate binary
  // representation during bit operations.
  uint8_t bytes[4];
  if (endianness == Endianness::LittleEndian) {
    bytes[3] = static_cast<uint8_t>(num >> 24) & 0xFF;
    bytes[2] = static_cast<uint8_t>(num >> 16) & 0xFF;
    bytes[1] = static_cast<uint8_t>(num >> 8) & 0xFF;
    bytes[0] = static_cast<uint8_t>(num) & 0xFF;
  } else {
    bytes[0] = static_cast<uint8_t>(num >> 24) & 0xFF;
    bytes[1] = static_cast<uint8_t>(num >> 16) & 0xFF;
    bytes[2] = static_cast<uint8_t>(num >> 8) & 0xFF;
    bytes[3] = static_cast<uint8_t>(num) & 0xFF;
  }
  // Append the converted bytes to the fileData vector.
  for (size_t i = 0; i < 4; i++)
    fileData.push_back(bytes[i]);
}

// Converts a 16-bit integer to two bytes based on endianness.
// Params:
//   fileData: Vector containing the raw binary data.
//   num: A 16-bit integer prepared for convertion.
//   endianness: Specifies the byte order (LittleEndian or BigEndian).
template <typename T, size_t N>
void Audio<T, N>::i16ToTwoBytes(std::vector<uint8_t> &fileData, int16_t num,
                                Endianness endianness) {
  // Use uint8_t to prevent sign extension and maintain accurate binary
  // representation during bit operations.
  uint8_t bytes[2];
  if (endianness == Endianness::LittleEndian) {
    bytes[1] = static_cast<uint8_t>(num >> 8) & 0xFF;
    bytes[0] = static_cast<uint8_t>(num) & 0xFF;
  } else {
    bytes[0] = static_cast<uint8_t>(num >> 8) & 0xFF;
    bytes[1] = static_cast<uint8_t>(num) & 0xFF;
  }
  // Append the converted bytes to the fileData vector.
  fileData.push_back(bytes[0]);
  fileData.push_back(bytes[1]);
}

// Converts an audio sample to a 8-bit PCM format (one byte).
// Params:
//   sample: A floating-point value representing the audio sample.
// Returns:
//   An 8-bit unsigned integer representing the sample as one byte.
template <typename T, size_t N> uint8_t Audio<T, N>::sampleToOneByte(T sample) {
  if (std::isnan(sample)) {
    // Handle corner case for NaN (Not a Number). Reset NaN to 1.
    sample = static_cast<T>(1.);
  } else {
    // Restricts sample value in range [-1.0, 1.0].
    sample = std::min(sample, static_cast<T>(1.));
    sample = std::max(sample, static_cast<T>(-1.));
  }
  // Converts a normalized floating-point audio sample to the [0, 255] range.
  sample = (sample + static_cast<T>(1.)) / static_cast<T>(2.);
  return static_cast<uint8_t>(sample * 255.);
}

// Converts an audio sample to a 16-bit PCM format (two bytes).
// Params:
//   sample: A floating-point value representing the audio sample.
// Returns:
//   A 16-bit signed integer representing the sample as two bytes.
template <typename T, size_t N> int16_t Audio<T, N>::sampleToI16(T sample) {
  if (std::isnan(sample)) {
    // Handle corner case for NaN (Not a Number). Reset NaN to 1.
    sample = static_cast<T>(1.);
  } else {
    // Restricts sample value in range [-1.0, 1.0].
    sample = std::min(sample, static_cast<T>(1.));
    sample = std::max(sample, static_cast<T>(-1.));
  }
  // Converts a normalized floating-point audio sample to the [-32767, 32767]
  // range.
  return static_cast<int16_t>(sample * 32767.);
}
} // namespace dap

#endif // FRONTEND_INTERFACES_BUDDY_DAP_AUDIOCONTAINER
