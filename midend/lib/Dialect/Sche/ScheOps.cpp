//===- ScheOps.cpp - sche Dialect Ops -----------------------------*- C++ -*-===//
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
// This file defines operations in the sche dialect.
//
//===----------------------------------------------------------------------===//

#include "Sche/ScheOps.h"
#include "Sche/ScheDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace buddy::sche;

namespace buddy::sche {

    using BodyBuilderFn = mlir::function_ref<void(OpBuilder&, Location, ValueRange)>;
    
    void OnDeviceOp::build(OpBuilder& builder, OperationState& result, llvm::StringRef targetId,
                         llvm::StringRef targetConfig, TypeRange resultTypes, ValueRange args,
                         BodyBuilderFn bodyBuilder, Type asyncTokenType, ValueRange asyncDependencies) {
        result.addAttribute(getTargetIdAttrName(result.name), builder.getStringAttr(targetId));
        result.addAttribute(getTargetConfigAttrName(result.name), builder.getStringAttr(targetConfig));
        result.addOperands(asyncDependencies);
        result.addOperands(args);

        if (asyncTokenType)
            result.types.push_back(builder.getType<AsyncTokenType>());

        Region* bodyRegion = result.addRegion();
        bodyRegion->push_back(new Block());
        auto& bodyBlock = bodyRegion->front();
        result.addTypes(resultTypes);

        if (bodyBuilder) {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(&bodyBlock);
            bodyBuilder(builder, result.location, bodyBlock.getArguments());
        } else {
            llvm_unreachable("no body builder given for OnDevice op builder.");
        }

        SmallVector<int32_t, 8> operandSegmentSizes(2, 1);
        operandSegmentSizes.front() = asyncDependencies.size();
        operandSegmentSizes.back() = args.size();
        result.addAttribute(getOperandSegmentSizeAttr(),
                            builder.getDenseI32ArrayAttr(operandSegmentSizes));

        SmallVector<int32_t, 8> resultSegmentSizes(2, 1);
        resultSegmentSizes.front() = asyncTokenType ? 1 : 0;
        resultSegmentSizes.back() = resultTypes.size();
        result.addAttribute(getResultSegmentSizeAttr(),
                            builder.getDenseI32ArrayAttr(resultSegmentSizes));
    }

    /// Parses an optional list of async operands with an optional leading keyword.
    /// (`async`)? (`[` ssa-id-list `]`)?
    ///
    /// This method is used by the tablegen assembly format for async ops as well.
    static ParseResult parseAsyncDependencies(
        OpAsmParser &parser, Type &asyncTokenType,
        SmallVectorImpl<OpAsmParser::UnresolvedOperand> &asyncDependencies) {
    auto loc = parser.getCurrentLocation();
    if (succeeded(parser.parseOptionalKeyword("async"))) {
        if (parser.getNumResults() == 0)
        return parser.emitError(loc, "needs to be named when marked 'async'");
        asyncTokenType = parser.getBuilder().getType<AsyncTokenType>();
    }
    return parser.parseOperandList(asyncDependencies,
                                    OpAsmParser::Delimiter::OptionalSquare);
    }

    //  Prints optional async dependencies with its leading keyword.
    //    (`async`)? (`[` ssa-id-list `]`)?
    // Used by the tablegen assembly format for several async ops.
    static void printAsyncDependencies(OpAsmPrinter &printer, Operation *op,
                                    Type asyncTokenType,
                                    OperandRange asyncDependencies) {
    if (asyncTokenType)
        printer << "async";
    if (asyncDependencies.empty())
        return;
    if (asyncTokenType)
        printer << ' ';
    printer << '[';
    llvm::interleaveComma(asyncDependencies, printer);
    printer << ']';
    }

    void OnDeviceOp::print(OpAsmPrinter &p){
        if (getAsyncToken()) {
            p << " async ";
            if (!getAsyncDependencies().empty())
            p << " [" << getAsyncDependencies() << "] ";
        }
        p << "(";
        p << getInnerOperands();
        p << ")";
        ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
        elidedAttrs.push_back("operand_segment_sizes");
        elidedAttrs.push_back("result_segment_sizes");
        p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
        p << ' ';
        p.printFunctionalType(getInnerOperands().getTypes(), getInnerResults().getTypes());
        p << ' ';
        p.printRegion(getRegion());
        elidedAttrs.push_back("operand_segment_sizes");

    }
    
    //operation ::= `sche.on_device` (`async` `[` ssa-id-list `]`)? `(`$operands`)` attr-dict functional-type(operands, results) $region
    ::mlir::ParseResult OnDeviceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
    ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> asyncDependencies;
    ::llvm::SMLoc asyncDependenciesOperandsLoc;
    (void)asyncDependenciesOperandsLoc;
    ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> innerOperandsOperands;
    ::llvm::SMLoc innerOperandsOperandsLoc;
    (void)innerOperandsOperandsLoc;
    ::llvm::ArrayRef<::mlir::Type> innerOperandsTypes;
    ::llvm::ArrayRef<::mlir::Type> innerResultsTypes;
    std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
    ::mlir::Type asyncTokenType;
    if (succeeded(parser.parseOptionalKeyword("async"))) {
        result.types.push_back(parser.getBuilder().getType<AsyncTokenType>());
    }
    if (failed(
          parseAsyncDependencies(parser, asyncTokenType, asyncDependencies)) ||
      parser.resolveOperands(asyncDependencies, asyncTokenType,
                             result.operands))
    return failure();
    if (parser.parseLParen())
        return ::mlir::failure();
    innerOperandsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(innerOperandsOperands))
        return ::mlir::failure();
    if (parser.parseRParen())
        return ::mlir::failure();
    if (parser.parseOptionalAttrDict(result.attributes))
        return ::mlir::failure();
    ::mlir::FunctionType innerOperands__innerResults_functionType;
    if (parser.parseType(innerOperands__innerResults_functionType))
        return ::mlir::failure();
    innerOperandsTypes = innerOperands__innerResults_functionType.getInputs();
    innerResultsTypes = innerOperands__innerResults_functionType.getResults();
    if (parser.parseRegion(*regionRegion))
        return ::mlir::failure();
    result.addRegion(std::move(regionRegion));
    result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(asyncDependencies.size()), static_cast<int32_t>(innerOperandsOperands.size())}));
    result.addAttribute("result_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(parser.getNumResults()) - static_cast<int32_t>(innerResultsTypes.size()), static_cast<int32_t>(innerResultsTypes.size())}));
    result.addTypes(innerResultsTypes);
    if (parser.resolveOperands(innerOperandsOperands, innerOperandsTypes, innerOperandsOperandsLoc, result.operands))
        return ::mlir::failure();
    return ::mlir::success();
    }

} //namespace


#define GET_OP_CLASSES
#include "Sche/ScheOps.cpp.inc"
