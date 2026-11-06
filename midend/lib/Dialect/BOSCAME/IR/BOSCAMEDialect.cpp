//===- BOSCAMEDialect.cpp - MLIR BOSCAME dialect implementation -------===//
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
// Dialect registration plus the per-operation half of the SPM/DMA contract.
//
// TableGen-generated attr/type/op classes are included here.  Local verifiers
// cover `bosc_ame.spm.view` and `bosc_ame.dma.start` parameters (banks,
// endpoints, static transfer length, alignment) and the DMA memory-effect
// interface.  Cross-operation rules (token lifetime, channel occupancy,
// aliasing) live in `SpmDmaCheck.cpp` / `verifySpmDmaContract`.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect/BOSCAME/BOSCAMEDialect.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/DmaAlignment.h"
#include "Dialect/BOSCAME/SpmDma.h"

using namespace mlir;
using namespace buddy::boscame;
using namespace buddy::boscame::spmdma;

#define GET_ATTRDEF_CLASSES
#include "BOSCAME/BOSCAMEAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "BOSCAME/BOSCAMETypes.cpp.inc"

#include "BOSCAME/BOSCAMEDialect.cpp.inc"

#define GET_OP_CLASSES
#include "BOSCAME/BOSCAME.cpp.inc"

//===----------------------------------------------------------------------===//
// SPM storage space attribute
//===----------------------------------------------------------------------===//

LogicalResult SPMAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              int64_t bank) {
  if (bank < 0 || bank >= kSpmBankCount)
    return emitError() << "SPM bank must be in [0, " << (kSpmBankCount - 1)
                       << "], got " << bank;
  return success();
}

//===----------------------------------------------------------------------===//
// DMA request token
//===----------------------------------------------------------------------===//

LogicalResult DMATokenType::verify(function_ref<InFlightDiagnostic()> emitError,
                                   int64_t channel) {
  if (channel < 0 || channel >= kDmaChannelCount)
    return emitError() << "DMA channel must be in [0, "
                       << (kDmaChannelCount - 1) << "], got " << channel;
  return success();
}

//===----------------------------------------------------------------------===//
// SPM / DMA verifiers
//===----------------------------------------------------------------------===//

// A byte range inside one SPM bank, expressed relative to the start of that
// bank.  Only ranges that are provably bank-anchored may be compared with each
// other; an opaque `memref` that merely *carries* the SPM storage space is not
// necessarily anchored at the bank base.
struct BankAnchoredRange {
  int64_t bank;
  int64_t begin;
  int64_t end;
};

// Derive the bank-anchored byte range that a transfer of `bytes` bytes
// starting at `value` occupies, or `std::nullopt` when the value cannot be
// traced back to a bank-anchored origin.  The two supported origins are the
// result of a `bosc_ame.spm.view` (whose `offset` is bank-relative by
// construction) and an SPM block argument, which the workspace contract pins
// to the start of its bank.
static std::optional<BankAnchoredRange> getBankAnchoredRange(Value value,
                                                             int64_t bytes) {
  auto type = dyn_cast<MemRefType>(value.getType());
  auto bank = getSpmBank(type);
  auto origin = getStaticPointerOrigin(value);
  auto offset = getStaticDescriptorOffset(value);
  unsigned elementBytes =
      type ? getSpmElementByteWidth(type.getElementType()) : 0;
  if (!bank || !origin || !offset || !elementBytes || bytes <= 0)
    return std::nullopt;
  int64_t begin, end;
  if (llvm::MulOverflow(*offset, int64_t(elementBytes), begin) ||
      llvm::AddOverflow(begin, origin->byteShift, begin) || begin < 0 ||
      llvm::AddOverflow(begin, bytes, end))
    return std::nullopt;

  int64_t base = 0;
  std::optional<int64_t> capacity;
  if (auto view = origin->root.getDefiningOp<SpmViewOp>()) {
    if (view.getBank() != *bank ||
        view.getOffset() > uint64_t(kSpmBankCapacityBytes))
      return std::nullopt;
    base = static_cast<int64_t>(view.getOffset());
    capacity =
        getContiguousZeroOffsetByteSize(cast<MemRefType>(view.getType()));
  }
  // A block argument counts as bank-anchored only when it is exactly one
  // complete bank: the workspace contract maps each bank used by a kernel to
  // one such parameter.  A narrower SPM memref argument has no statically known
  // position inside its bank.
  else if (isa<BlockArgument>(origin->root)) {
    auto memref = dyn_cast<MemRefType>(origin->root.getType());
    if (!memref)
      return std::nullopt;
    capacity = getContiguousZeroOffsetByteSize(memref);
    if (getSpmBank(memref) != bank || !capacity ||
        *capacity != kSpmBankCapacityBytes)
      return std::nullopt;
  }
  if (!capacity || end > *capacity || llvm::AddOverflow(begin, base, begin) ||
      llvm::AddOverflow(end, base, end) || end > kSpmBankCapacityBytes)
    return std::nullopt;
  return BankAnchoredRange{*bank, begin, end};
}

// Verify the static part of an `bosc_ame.spm.view`.  Everything that is a
// property of the operation itself lives here; the cross-operation rules
// (tokens, channel occupancy, buffer conflicts) live in
// `bosc-ame-spm-dma-check`.
LogicalResult SpmViewOp::verify() {
  int64_t bank = getBank();
  if (bank < 0 || bank >= kSpmBankCount)
    return emitOpError() << "bank must be in [0, " << (kSpmBankCount - 1)
                         << "], got " << bank;

  auto viewType = dyn_cast<MemRefType>(getView().getType());
  if (!viewType)
    return emitOpError("result must be a ranked memref");

  auto space = dyn_cast_or_null<SPMAttr>(viewType.getMemorySpace());
  if (!space)
    return emitOpError() << "result must carry the #bosc_ame.spm<bank> storage "
                            "space, got "
                         << getView().getType();
  if (space.getBank() != bank)
    return emitOpError() << "bank " << bank
                         << " does not match the result storage space "
                         << space;

  if (!viewType.hasStaticShape())
    return emitOpError("result must have a static shape");

  // The size of the view comes from the result type, never from a second
  // redundant size operand: two independent descriptions of the same byte range
  // could disagree.
  if (getSpmElementByteWidth(viewType.getElementType()) == 0)
    return emitOpError() << "element type " << viewType.getElementType()
                         << " is not addressable in SPM; supported element "
                            "types are i8/i16/i32/i64/f16/bf16/f32/f64";

  std::optional<int64_t> sizeBytes = getContiguousZeroOffsetByteSize(viewType);
  if (!sizeBytes)
    return emitOpError("result must be contiguous with a zero descriptor "
                       "offset; a strided or otherwise shifted SPM view has no "
                       "single byte range");
  if (*sizeBytes <= 0)
    return emitOpError("result must not be empty");
  if (*sizeBytes > kSpmBankCapacityBytes)
    return emitOpError() << "view of " << *sizeBytes
                         << " bytes does not fit in a " << kSpmBankCapacityBytes
                         << "-byte SPM bank";

  uint64_t offset = getOffset();
  if (static_cast<int64_t>(offset) < 0)
    return emitOpError() << "offset must be non-negative, got "
                         << static_cast<int64_t>(offset);
  // `requiredBytes` is <= 128 KiB, so this addition cannot overflow.
  uint64_t requiredBytes = static_cast<uint64_t>(*sizeBytes);
  if (offset > static_cast<uint64_t>(kSpmBankCapacityBytes) - requiredBytes)
    return emitOpError() << "byte range [" << offset << ", "
                         << (offset + requiredBytes) << ") exceeds the "
                         << kSpmBankCapacityBytes << "-byte SPM bank " << bank;

  return success();
}

// Verify the static part of a DMA request.  The verifier answers, without
// looking at any other operation, whether the request is a well-formed
// description of one contiguous transfer of the patch7 DMA engines.
LogicalResult DmaStartOp::verify() {
  int64_t channel = getChannel();
  if (channel < 0 || channel >= kDmaChannelCount)
    return emitOpError() << "channel must be in [0, " << (kDmaChannelCount - 1)
                         << "], got " << channel;

  auto tokenType = dyn_cast<DMATokenType>(getToken().getType());
  if (!tokenType)
    return emitOpError() << "result must be !bosc_ame.dma_token<channel>, got "
                         << getToken().getType();
  if (tokenType.getChannel() != channel)
    return emitOpError() << "result token channel " << tokenType.getChannel()
                         << " does not match the request channel " << channel;

  auto srcType = dyn_cast<MemRefType>(getSrc().getType());
  auto dstType = dyn_cast<MemRefType>(getDst().getType());
  if (!srcType)
    return emitOpError() << "src must be a ranked memref, got "
                         << getSrc().getType();
  if (!dstType)
    return emitOpError() << "dst must be a ranked memref, got "
                         << getDst().getType();

  std::optional<DmaEndpoint> src = getDmaEndpoint(srcType);
  std::optional<DmaEndpoint> dst = getDmaEndpoint(dstType);
  if (!src)
    return emitOpError() << "src must be a 1-D byte-addressable memref with a "
                            "static shape and unit stride, got "
                         << srcType;
  if (!dst)
    return emitOpError() << "dst must be a 1-D byte-addressable memref with a "
                            "static shape and unit stride, got "
                         << dstType;
  if (!src->isSpm() && !dst->isSpm())
    return emitOpError("at least one endpoint must be SPM storage; DDR to DDR "
                       "transfers are not part of the v1 contract");

  std::optional<int64_t> bytesValue = getConstantIntValue(getBytes());
  if (!bytesValue)
    return emitOpError(
        "v1 requires a statically known transfer length: `bytes` "
        "must fold to a constant number of bytes");
  int64_t bytes = *bytesValue;
  if (bytes <= 0)
    return emitOpError() << "bytes must be positive, got " << bytes;
  if (bytes % kDmaAlignmentBytes != 0)
    return emitOpError() << "bytes must be a multiple of the "
                         << kDmaAlignmentBytes << "-byte DMA granularity, got "
                         << bytes;

  if (bytes > src->byteLength)
    return emitOpError() << "transfer of " << bytes
                         << " bytes exceeds the src byte range of "
                         << src->byteLength << " bytes";
  if (bytes > dst->byteLength)
    return emitOpError() << "transfer of " << bytes
                         << " bytes exceeds the dst byte range of "
                         << dst->byteLength << " bytes";

  for (const DmaEndpoint *endpoint : {&*src, &*dst}) {
    if (!endpoint->isSpm())
      continue;
    if (endpoint->hasStaticOffset &&
        (endpoint->byteOffset < 0 ||
         endpoint->byteOffset > kSpmBankCapacityBytes - bytes))
      return emitOpError() << "byte range [" << endpoint->byteOffset << ", "
                           << (endpoint->byteOffset + bytes)
                           << ") leaves SPM bank " << *endpoint->bank;
    if (endpoint->hasStaticOffset && !isStaticallyAlignedEndpoint(*endpoint))
      return emitOpError()
             << "SPM endpoint must be 64 B aligned, but its descriptor offset "
             << endpoint->byteOffset << " is not";
  }

  // The descriptor offset is only one of the three contributions to an
  // effective address.  A `memref.view` shift moves the base pointer while
  // keeping that offset at zero, so every shift on the endpoint's view chain
  // has to be either static and aligned or provably a multiple of the DMA
  // granularity; otherwise the alignment of the transferred range is a guess.
  if (failed(verifyEndpointPointerAlignment(getOperation(), getSrc(), "src",
                                            kDmaAlignmentBytes)) ||
      failed(verifyEndpointPointerAlignment(getOperation(), getDst(), "dst",
                                            kDmaAlignmentBytes)))
    return failure();

  // A DDR base address is not a compile-time constant.  The module-level ABI
  // assertion records 64 B aligned arguments. Only the stronger nonalias
  // assertion supplies pairwise disjointness to the lifetime checker.
  auto abi =
      getOperation()->getParentOfType<ModuleOp>()->getAttrOfType<StringAttr>(
          kDmaAbiAttrName);
  if ((!src->isSpm() || !dst->isSpm()) &&
      (!abi || (abi.getValue() != kDmaAbiAlignedNonAliasing &&
                abi.getValue() != kDmaAbiAligned)))
    return emitOpError()
           << "a DDR endpoint cannot be proven aligned and non-overlapping; "
              "annotate the module with `"
           << kDmaAbiAttrName << " = \"" << kDmaAbiAlignedNonAliasing << "\"`";

  if (src->isSpm() && dst->isSpm() && *src->bank == *dst->bank) {
    std::optional<BankAnchoredRange> srcRange =
        getBankAnchoredRange(getSrc(), bytes);
    std::optional<BankAnchoredRange> dstRange =
        getBankAnchoredRange(getDst(), bytes);
    if (!srcRange || !dstRange)
      return emitOpError()
             << "cannot prove that the src and dst ranges in SPM bank "
             << *src->bank
             << " are disjoint: cannot statically bound an endpoint within "
                "its bosc_ame.spm.view or full-bank SPM block argument";
    if (srcRange->begin < dstRange->end && dstRange->begin < srcRange->end)
      return emitOpError() << "src range [" << srcRange->begin << ", "
                           << srcRange->end << ") and dst range ["
                           << dstRange->begin << ", " << dstRange->end
                           << ") overlap in SPM bank " << *src->bank;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DMA memory effects
//===----------------------------------------------------------------------===//

namespace {
// The DMA engines as a side-effect resource.  Both `dma.start` and `dma.wait`
// write this resource, so no optimization may reorder, eliminate or duplicate
// a transfer or its completion.  Volatile MMIO alone would not be enough: the
// effect interface is what keeps the request alive and its status read where
// the program put it.
struct DmaDeviceResource
    : public SideEffects::Resource::Base<DmaDeviceResource> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DmaDeviceResource)
  StringRef getName() const final { return "bosc_ame.dma"; }
};
} // namespace

void DmaStartOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // The request reads the source range and writes the destination range, but
  // the transfer is asynchronous: these effects describe the window that starts
  // here and is closed by the matching `dma.wait`.  Because no interface can
  // express "written later, by the device", the op also keeps the conservative
  // default-resource barrier below.
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
  effects.emplace_back(MemoryEffects::Write::get(), DmaDeviceResource::get());
}

void DmaWaitOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // Waiting publishes the destination range written by the device without the
  // compiler knowing which range that is; only the paired `dma.start` does.
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
  effects.emplace_back(MemoryEffects::Write::get(), DmaDeviceResource::get());
}

void BOSCAMEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "BOSCAME/BOSCAME.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "BOSCAME/BOSCAMETypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "BOSCAME/BOSCAMEAttrs.cpp.inc"
      >();
}
