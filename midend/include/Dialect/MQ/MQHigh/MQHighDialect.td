//===- MQHighDialect.td - MQHigh dialect definition --------*- tablegen -*-===//
//
// This file defines the MQHigh dialect for the MoonQuest accelerator.
//
//===----------------------------------------------------------------------===//

#ifndef BUD_MQHIGH_DIALECT
#define BUD_MQHIGH_DIALECT

// include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// MQHigh Dialect Definition.
//===----------------------------------------------------------------------===//

def MQHigh_Dialect : Dialect {
  let name = "mqhigh";
  let summary = "A high-level dialect for the ONNX MQ accelerator ISA.";
  let cppNamespace = "::buddy::mqhigh";

  let hasConstantMaterializer = 0;
  let hasNonDefaultDestructor = 0;
  let hasOperationAttrVerify = 0;
  let hasRegionArgAttrVerify = 0;
  let hasRegionResultAttrVerify = 0;
  let hasOperationInterfaceFallback = 0;
  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
  let hasCanonicalizer = 0;
  let isExtensible = 0;
  let usePropertiesForAttributes = 1;
}

//===----------------------------------------------------------------------===//
// Base MQHigh Operation Definition.
//===----------------------------------------------------------------------===//

class MQHigh_Op<string mnemonic, list<Trait> traits = []> :
    Op<MQHigh_Dialect, mnemonic, traits>;

#endif // BUD_MQHIGH_DIALECT
