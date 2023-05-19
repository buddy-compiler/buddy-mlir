//===- DynamicVLOpInterface.td -----------------------------*- tablegen -*-===//
//
// This is the definition file for the dynamic vector length interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_DIALECT_VECTOREXP_DYNAMICVLOPINTERFACES_TD
#define BUDDY_DIALECT_VECTOREXP_DYNAMICVLOPINTERFACES_TD

include "mlir/IR/OpBase.td"

def DynamicVLOpInterface : OpInterface<"DynamicVLOpInterface"> {
  let description = [{
    The 'DynamicVLOpInterface' defines an operation that can apply dynamic
    vector length.
  }];
  let cppNamespace = "::buddy::vector_exp";
  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the dynamic VL value.",
      /*retTy=*/"mlir::TypedValue<mlir::IntegerType>",
      /*methodName=*/"getVLValue",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"">,
    InterfaceMethod<
      /*desc=*/"Build function for dynamic VL operand.",
      /*retTy=*/"void",
      /*methodName=*/"buildVL",
      /*args=*/(ins "mlir::OperationState &":$odsState, "mlir::Value":$vl),
      /*methodBody=*/"",
      /*defaultImplementation=*/"odsState.addOperands(vl);">,
    InterfaceMethod<
      /*desc=*/"Parser for dynamic VL operand.",
      /*retTy=*/"mlir::ParseResult",
      /*methodName=*/"parseVL",
      /*args=*/(ins "mlir::OpAsmParser &":$parser,
                    "mlir::OperationState &":$result),
      /*methodBody=*/"",
      /*defaultImplementation=*/"">,
    InterfaceMethod<
      /*desc=*/"Printer for dynamic VL operand.",
      /*retTy=*/"void",
      /*methodName=*/"printVL",
      /*args=*/(ins "mlir::OpAsmPrinter &":$_odsPrinter),
      /*methodBody=*/"",
      /*defaultImplementation=*/"">,
  ];
}

#endif // BUDDY_DIALECT_VECTOREXP_DYNAMICVLOPINTERFACES_TD
