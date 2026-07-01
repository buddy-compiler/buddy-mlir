# Overall design process developed by BOSCAME

[TOC]

## 1. Definition of dialect

### 1.1 BOSCAME.td

​	This `.td` (TableGen) file is written based on the MLIR ODS (Operation Definition Specification) framework and is mainly used to define the **BOSCAME dialect**. Its core function is to abstract the underlying hardware instructions of the **RISC-V matrix extension** into computation operations (Operations) in MLIR.

```c++
def BOSCAME_Dialect : Dialect {
  let name = "bosc_ame";
  let cppNamespace = "::buddy::boscame";
  let description = [{
    The BOSCAME dialect provides operations for the RISC-V Matrix Extension (RVA23 profile).
    BOSCAME provides matrix multiply-accumulate instructions with tile-based computation,
    supporting various data types and widening modes for AI/ML workloads.

    Key features:
    - Tile-based matrix registers with configurable dimensions
    - Support for int4/int8/int16/int32/int64 integer types
    - Support for fp8/fp16/fp32/fp64 floating-point types
    - Widening matrix multiplication (2x, 4x, 8x output width)
    - Saturating arithmetic options
  }];
}
```

- Registered the `bosc_ame` dialect and its corresponding C++ namespace `::buddy::boscame`, and gave some related description.

- Definition of the instruction formats for the relevant dialect specs: status configuration instructions, load/store instructions, data move instructions, element-wise operation instructions, matrix multiply-add instructions, and support for data generics in each instruction.
- Building a bridge to downgrade to LLVM IR.

```c++
class BOSCAME_IntrOpBase<string mnemonic, int numRes = 0, list<Trait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/BOSCAME_Dialect,
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_buddy_bosc_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[],
                  /*list<int> overloadedOperands=*/[],
                  /*list<Trait> traits=*/traits,
                  /*int numResults=*/numRes>;
```

​	The classes in the latter part of the document with the `_IntrOp` suffix (like `BOSCAME_Msettype_IntrOp`, `BOSCAME_MmaMm_IntrOp`) are specifically prepared for LLVM translation. They map the higher-level MLIR operations one-to-one to the lower-level LLVM intrinsics (corresponding to the `riscv_buddy_bosc_` naming convention).

### 1.2 BOSCAMEDialect.h

​	This file is the **main entry and core declaration file** for the entire BOSCAME dialect in the C++ codebase. It acts as the key bridge that really brings those TableGen definitions into the Buddy-MLIR compiler project.

```c++
#include "BOSCAME/BOSCAMEDialect.h.inc"
```

​	Automatically reads the `BOSCAME.td` file and generates a lot of C++ template code and class definitions, which are stored in the temporary file `BOSCAMEDialect.h.inc`.

### 1.3 BOSCAMEOps.h

​	All the **specific instruction 'operation manuals'** within the dialect will take the hundreds of matrix instructions I defined in `BOSCAME.td` using TableGen syntax (such as `msettype`, `mma.mm`, `mlae8.m`, etc.) and instantiate them into real C++ classes, exposing them to the compiler project.

```c++
#define GET_OP_CLASSES
#include "BOSCAME/BOSCAME.h.inc"
```

​	When MLIR's TableGen tool processes my `.td` file, it generates a huge `.inc` file. This file contains all sorts of things, including dialect definitions, type definitions, and operation class declarations. When writing the conversion and optimisation passes for Buddy-MLIR later, this file is essential to reference.

### 1.4 Transform.h

​	`Transform.h` file is basically the compiler's 'core translator' (Lowering / Conversion) declaration file. In the MLIR compilation process, high-level matrix instructions (like `boscame.mma.mm`) ultimately have to be translated (Lowering) into low-level instructions that LLVM can recognise (that is, the `intr.msettype` and other LLVM Intrinsics I defined in the `.td` files) before RISC-V machine code can finally be generated.

​	This header file is meant to declare the interface for **how to carry out this translation**. Let's go through its core parts one by one:

```c++
void populateBOSCAMELegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                  RewritePatternSet &patterns);
```

​	This function is responsible for registering all the **'conversion rules (Patterns)'** into the **rule set (`RewritePatternSet`)**.

```c++
void configureBOSCAMELegalizeForExportTarget(LLVMConversionTarget &target);
```

​	This is a verification mechanism. It's used to set up the conversion target, telling the MLIR framework which instructions will be legal and which will be illegal once the translation process is done.

```c++
std::unique_ptr<mlir::Pass> createLegalizeForLLVMExportPass();
```

​	As a function, create a conversion Pass object for mlir, then call this function to load the Pass into the `PassManager`. When the pipeline reaches this step, it will call the previous two functions, go through the entire program code, and replace all BOSCAME instructions with LLVM instructions.

### 1.5 CMakeLists.txt

This is a **CMake build script**:

```c++
add_mlir_dialect(BOSCAME bosc_ame)
```

​	Register the dialect and define the namespace prefix as `bosc_ame`, automatically look for the corresponding `BOSCAME.td` file, and use the `mlir-tblgen` tool to generate the C++ declarations and definitions for the dialect (like the `BOSCAMEDialect.h.inc` and `BOSCAME.h.inc` included in the `.h` files earlier).

```c++
add_mlir_doc(BOSCAME BOSCAME Dialects/ -gen-dialect-doc)
```

​	Automatically generate dialect development documentation, extract the `summary` and `description` I wrote in `BOSCAME.td`, and automatically create a Markdown document in the `Dialects/` directory with the commands I defined and their function descriptions.

```c++
set(LLVM_TARGET_DEFINITIONS BOSCAME.td)
mlir_tablegen(BOSCAMEConversions.inc -gen-llvmir-conversions)
```

​	Generate conversion code for lowering to LLVM IR, directly corresponding to those classes with the `_IntrOp` suffix (Intrinsic Operations) that I wrote in the latter part of the `BOSCAME.td` file. Generate a C++ file named `BOSCAMEConversions.inc`. This generated file should contain **pattern matching and rewrite logic** that can automatically translate my BOSCAME MLIR instructions one-to-one into underlying LLVM inline functions (LLVM Intrinsics).

```c++
add_public_tablegen_target(BuddyBOSCAMEConversionsIncGen)
```

​	Manage build dependencies to ensure that `.inc` files are generated before `.cpp` files are compiled. By defining this Target, a C++ library can declare a dependency on it in CMake, ensuring a smooth compilation order without getting 'header file not found' errors.

File structure:

```markdown
midend/
├── include/Dialect/BOSCAME/
│   ├── CMakeLists.txt         # TableGen configuration and code generation script
│   ├── BOSCAME.td             # Dialect TableGen definition (hardware instructions and type abstraction)
│   ├── BOSCAMEDialect.h       # Dialect registration header file (main entry)
│   ├── BOSCAMEOps.h           # Header file for operations (C++ class for specific matrix instructions)
│   └── Transform.h            # Header file for downgrade conversion (Pass interface for converting to LLVM IR)
```

## 2. Basic command definition

### 2.1 Resources corresponding to the underlying hardware

​	The `RISCVBuddyExt.td` file is the 'hardware resource registry' and 'master switch' I built in the LLVM backend for this RISC-V matrix extension, defining exactly 'which physical hardware the aforementioned AME extension instructions can run on'.

​	The 'master switch' for AME extended architecture features, the content below registers a brand new CPU feature with LLVM's target architecture system, called `buddyext`; when compiling to the final C/C++ or MLIR code, you can activate this switch by passing a compilation flag (use `-mattr=+buddyext` in `llc`). The `let Predicates = [HasBuddyExt]` in the `RISCVInstrInfoBuddyBOSCExt.td` file is basically telling LLVM: '**only when we turn on this switch are we allowed to generate the matrix extension instructions I defined**'.

```c++
def FeatureBuddyExt
    : SubtargetFeature<"buddyext", "HasBuddyExt", "true",
                       "'BuddyExt' (Buddy RISC-V Extension)">;
def HasBuddyExt : Predicate<"Subtarget->hasBuddyExt()">,
                            AssemblerPredicate<(all_of FeatureBuddyExt),
                            "'BuddyExt' (Buddy RISC-V Extension)">;
```

​	Defining the matrix registers in the specified hardware, here I’ve hardcoded the new physical register file introduced by this hardware extension: 8 Tile registers for input (`tr0-tr7`) and 8 Accumulator registers for output accumulation (`acc0-acc7`); `HWEncoding` decides their 5-bit or 3-bit binary encoding in the final 32-bit machine code. When LLVM finally emits the machine code, it looks up this table to translate `tr1` to the binary `001`.

```c++
let Namespace = "RISCV" in {

// Base class for Tile Registers
class AMETileReg<bits<3> Enc, string n> : Register<n> {
  let HWEncoding{2-0} = Enc;
  let HWEncoding{4-3} = 0b00;  // Distinguish from accumulation registers
}

// Define 8 Tile Registers: tr0-tr7
def TR0 : AMETileReg<0, "tr0">;
def TR1 : AMETileReg<1, "tr1">;
def TR2 : AMETileReg<2, "tr2">;
def TR3 : AMETileReg<3, "tr3">;
def TR4 : AMETileReg<4, "tr4">;
def TR5 : AMETileReg<5, "tr5">;
def TR6 : AMETileReg<6, "tr6">;
def TR7 : AMETileReg<7, "tr7">;

// Base class for Accumulation Registers
class AMEAccReg<bits<3> Enc, string n> : Register<n> {
  let HWEncoding{2-0} = Enc;
  let HWEncoding{4-3} = 0b01;  // Distinguish from tile registers
}

// Define 8 Accumulation Registers: acc0-acc7
def ACC0 : AMEAccReg<0, "acc0">;
def ACC1 : AMEAccReg<1, "acc1">;
def ACC2 : AMEAccReg<2, "acc2">;
def ACC3 : AMEAccReg<3, "acc3">;
def ACC4 : AMEAccReg<4, "acc4">;
def ACC5 : AMEAccReg<5, "acc5">;
def ACC6 : AMEAccReg<6, "acc6">;
def ACC7 : AMEAccReg<7, "acc7">;

} // End Namespace = "RISCV"
```

​	Just defining independent registers isn't enough, LLVM's **Register Allocator** needs to know which registers belong to the same class and can be swapped with each other; that's why `AccReg` and `TileReg` are defined here. This is a strict constraint that ensures the compiler will never wrongly assign the result of a matrix multiplication to a `tr` register, keeping the program correct for the hardware design.

```c++
def TileReg : RegisterClass<"RISCV", [untyped], 256,
                            (add TR0, TR1, TR2, TR3, TR4, TR5, TR6, TR7)> {
  let Size = 256;  // Placeholder: actual MLEN is hardware-defined
}

def AccReg : RegisterClass<"RISCV", [untyped], 1024,
                           (add ACC0, ACC1, ACC2, ACC3, ACC4, ACC5, ACC6, ACC7)> {
  let Size = 1024;  // Placeholder: actual MLEN×AMUL is hardware-defined
}
```

​	Besides that, the most important thing is that the file `RISCVBuddyExt.td` acts as the top-level definition file, combining hardware resources and instruction set information into a complete Target description, which is then passed to LLVM's TableGen engine to generate the final backend C++ code.

```c++
include "RISCVInstrInfoBuddyBOSCExt.td"
```

### 2.2 Low-level hardware instruction mapping layer (LLVM ISel layer)

​	Turn these inline functions into actual RISC-V 0101 machine code and assembly that can run on the specified hardware through the `RISCVInstrInfoBuddyBOSCExt.td` file.

```c++
class RVInstBOSCAME32<dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<6> funct6;
  bit     fp;     // 0 for integer, 1 for floating-point (if supported in future)
  bit     sa;     // 0 for unsaturated, 1 for saturated
  bits<4> ms2;
  bit     sn;     // Whether the source operand is signed (for integers), 0 for floating-point
  bits<4> ms1;
  bits<3> eew;    // 0=int8, 1=int16, 2=int32, 3=int64, 7=int4
  bit     ma = 0b1;
  bits<4> md;
  bits<7> opcode = 0b1110111;

  let Inst{31-26} = funct6;
  let Inst{25}    = fp;
  let Inst{24}    = sa;
  let Inst{23-20} = ms2;
  let Inst{19}    = sn;
  let Inst{18-15} = ms1;
  let Inst{14-12} = eew;
  let Inst{11}    = ma;
  let Inst{10-7}  = md;
  let Inst{6-0} = 0b1110111;
}
```

​	Based on the instruction format in the Spec, different instruction operation classes are defined, precisely detailing the 32-bit instruction format for RISC-V, specifying the opcode (`opcode = 0b1110111`), function code (`funct6`), data width extension flag (`eew`), and the exact placement of each register (`rs1`, `rs2`, `md`) within the 32 bits. With this, LLVM's assembler and code emitter know how to pack the corresponding AME extension instructions I've written into the binary sequence in the final ELF or Bin files.

​	The `RISCVInstrInfoBuddyBOSCExt.td` file also defines **assembly 'printers' (Pseudo Instructions)**; through lots of `Pseudo` definitions (like `BOSC_AME_MMA_MM_PSEUDO`), it outputs readable assembly code. The hardware actually only recognises the designed hardware registers tr0~7 matrix registers and acc0~7 matrix registers, so by customising `BOSCAMETileIndex` and the assembly string (`AsmString = "mma.mm	acc$md, tr$ms1, tr$ms2"`), LLVM is forced to automatically add the `tr` (Tile Register) and `acc` (Accumulator Register) prefixes when generating `.s` assembly files, greatly improving the readability and debugging efficiency of the low-level assembly code.

```c++
let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 1, mayStore = 0,
    isCodeGenOnly = 1 in {
  def BOSC_AME_MLAE8_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlae8.m\ttr$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLAE16_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlae16.m\ttr$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLAE32_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlae32.m\ttr$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLAE64_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlae64.m\ttr$md, ($rs1), $rs2";
  }
}
```

​	The `RISCVInstrInfoBuddyBOSCExt.td` file also defines the instruction selection 'translation dictionary' (ISel Pattern Matching); this is the **most crucial** bit and really shows how the compilation flow works. By defining instruction modules, it turns MLIR code's high-level computations into the `int_riscv_buddy_bosc_mma_mm` instruction in LLVM IR, allowing LLVM's instruction selector (usually SelectionDAG) to do pattern matching. Once it matches, the `Pat<(int_riscv...), (BOSC_AME_MMA_MM_PSEUDO...)>` rule seamlessly replaces those abstract IR nodes with the RISC-V target machine instructions I defined above, complete with physical register constraints.

```c++
let Predicates = [HasBuddyExt] in {
  def : Pat<(int_riscv_buddy_bosc_mlae8_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLAE8_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlae16_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLAE16_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlae32_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLAE32_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlae64_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLAE64_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
}
```

### 2.3 Registering built-in functions (LLVM IR bridge layer)

​	The `IntrinsicsRISCVBuddyBOSCExt.td` file registers all the underlying intrinsic function signatures for BOSCAME instructions in LLVM, so the LLVM optimiser knows how to handle different instructions with different optimisations by tagging them with key side effect labels. For example, `IntrNoMem` tells the optimiser that MMA instructions don't touch memory, so they can be freely rearranged, while `IntrReadMem` protects Load instructions from being incorrectly eliminated. This is the foundation for keeping LLVM optimisations safe.

- Register official signature

​	In the code, each underlying matrix instruction's input and output types are explicitly declared to LLVM via `Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty], ...>`, for example:

```c++
class BOSC_AME_Load_Intr
    : Intrinsic<[], [llvm_i64_ty, llvm_ptr_ty, llvm_i64_ty],
                [IntrReadMem, IntrHasSideEffects,
                ImmArg<ArgIndex<0>>]>;

class BOSC_AME_Store_Intr
    : Intrinsic<[], [llvm_i64_ty, llvm_ptr_ty, llvm_i64_ty],
                [IntrWriteMem, IntrHasSideEffects,
                ImmArg<ArgIndex<0>>]>;
```

​	Since the relevant Load/Store instructions have different precisions for different numbers of operands, you can define the basic instructions using a class method and then just call the class to register operations with different precisions.

- Set up an optimiser guardrail

​	In the code above, the properties in the square brackets are the tags: **`IntrNoMem` (no memory access)**: applied to math operation instructions (like `mma`, `maddu`). This tells the LLVM optimiser: 'this instruction only operates on the relevant registers and will never read or write memory'. With this tag, LLVM’s dead code elimination and instruction reordering algorithms can freely move or optimise these matrix multiplications without worrying about breaking memory consistency; **`IntrReadMem` / `IntrWriteMem`**: applied to Load (`mlae8_m`) and Store (`msce8_m`) instructions, clearly warning the optimiser: 'this is a memory access operation, absolutely cannot be casually deleted, nor reordered across other memory instructions'; **`ImmArg<ArgIndex<0>>`**: tells the compiler that the 0th argument passed in must be a **compile-time constant (Immediate)**. This is because the hardware instruction’s matrix register index (0-7) must be determined at compile time and cannot be a dynamically calculated runtime variable.

​	At the same time, this part also needs the top-level file `IntrinsicsRISCV.td` to splice together the design and implementation of RISCV-related instructions, forming a complete workflow and ensuring that the implemented tags can receive targeted optimisation by the LLVM optimiser.

```c++
include "llvm/IR/IntrinsicsRISCVBuddyBOSCExt.td"
```

```markdown
llvm/
├── llvm/include/llvm/IR/
│   ├── IntrinsicsRISCV.td                 # RISC-V built-in function main entry (responsible for including various extensions)
│   └── IntrinsicsRISCVBuddyBOSCExt.td     # BOSC extends built-in function signatures, type checks and optimises side effect safeguards
├── llvm/lib/Target/RISCV/
│   ├── RISCVBuddyExt.td                   # Overall switch for architecture features and definition of the physical matrix register pool (TileReg/AccReg)
│   └── RISCVInstrInfoBuddyBOSCExt.td      # Low-level machine code encoding format, assembly pseudo-instruction printing and instruction selection matching dictionary
```

## 3. Dialect switch

### 3.1 BOSCAMEDialect.cpp

​	Create the dialect implementation file to actually instantiate the dialect and instructions I declared in the header file, and 'mount' them into the MLIR context so the compiler can truly recognise and use the `bosc_ame` matrix instructions I defined at runtime.

```c++
#include "BOSCAME/BOSCAMEDialect.cpp.inc"

#define GET_OP_CLASSES
#include "BOSCAME/BOSCAME.cpp.inc"
```

​	TableGen not only wrote the class declarations, but also all the getter/setter methods, verification logic (Verify) and parsing/printing logic (Parse/Print) in terms of concrete implementation. These two lines of code are equivalent to pasting thousands of lines of automatically generated C++ implementation into this file for compilation.

```c++
void BOSCAMEDialect::initialize() {
 addOperations<
#define GET_OP_LIST
#include "BOSCAME/BOSCAME.cpp.inc"
   >();
}
```

​	The initialisation entry for the dialect is responsible for mounting all elements belonging to the BOSCAME dialect (instructions, types, attributes, etc.) and registering all the matrix extension instructions I defined to the MLIR compiler in one go.

- CMakeLists.txt

```cmake
add_mlir_dialect_library(BuddyBOSCAME
 BOSCAMEDialect.cpp

 LINK_LIBS PUBLIC
 MLIRIR
)
```

​	**Encapsulate the core C++ implementation files for initializing the whole dialect and registering instructions into a standard, reusable software library**. This includes all the underlying data structures in MLIR intermediate representation like `Operation`, `Type`, `Attribute`, and `Value`, and declares their dependencies.

### 3.2 LegalizeForLLVMExport.cpp

​	Translate all BOSCAME instructions 1:1 perfectly into inline function (Intrinsics) calls that the underlying compiler (LLVM) can recognise. It's equivalent to converting the first part of my MLIR BOSCAME dialect through `LegalizeForLLVMExport.cpp` into the LLVM low-level instructions defined in the second part according to the AME extension Spec from OpenCore Institute, so that it can be compiled into target assembly files by LLVM's compilation tool llc for testing on the specified hardware or simulator.

- Core helper function

1. `getOrInsertIntrinsic`: Dynamically inserts LLVM function declarations. When you need to call a low-level instruction like `llvm.riscv.buddy.bosc.mma.mm`, the LLVM module must have the function's 'signature' first. This function checks if the current module has it declared, and if not, it generates one on the spot.

2. `extractPointerFromMemref`: strips out the memory pointer. In MLIR, memory is usually represented as a `memref` (basically a complex struct that contains pointer, shape, strides, and other info). But Load/Store instructions on RISC-V hardware only need a 'bare address'. This function is used to pull the raw pointer out of the `memref` struct, so it works with the corresponding hardware instructions.

```c++
static FlatSymbolRefAttr
getOrInsertIntrinsic(ConversionPatternRewriter &rewriter, ModuleOp module,
                     StringRef intrinsicName, LLVM::LLVMFunctionType funcType) {
  auto *ctx = rewriter.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(intrinsicName))
    return FlatSymbolRefAttr::get(ctx, intrinsicName);

  auto savedInsertionPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToEnd(module.getBody());
  LLVM::LLVMFuncOp::create(rewriter, module.getLoc(), intrinsicName, funcType,
                           LLVM::Linkage::External, false, LLVM::CConv::C);
  rewriter.restoreInsertionPoint(savedInsertionPoint);
  return FlatSymbolRefAttr::get(ctx, intrinsicName);
}

static Value extractPointerFromMemref(ConversionPatternRewriter &rewriter,
                                      Location loc, Value memref) {
  auto *ctx = rewriter.getContext();
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  auto i64Type = IntegerType::get(ctx, 64);
  Value idx =
      memref::ExtractAlignedPointerAsIndexOp::create(rewriter, loc, memref);
  Value i64Val = arith::IndexCastOp::create(rewriter, loc, i64Type, idx);
  Value ptr = LLVM::IntToPtrOp::create(rewriter, loc, ptrType, i64Val);
  return ptr;
}
```

- General downgrade template (Lowering Patterns)

​	Used **C++ templates**, inherited from `ConvertOpToLLVMPattern`, to get operands (like register indices `md`, `ms1`, `ms2`) $\rightarrow$ Convert them into LLVM constants or values $\rightarrow$Create a function call to the underlying Intrinsic (`LLVM::CallOp`) $\rightarrow$ **Delete the original high-level MLIR instruction (`rewriter.eraseOp(op)`)**. It's a standard Pass definition implementation in the MLIR repo, basically a Pattern Rewriting.

- Command mapping registration

​	By defining the function `populateBOSCAMELegalizeForLLVMExportPatterns`, each MLIR operator defined in the `BOSCAME.td` file (like `<MadduMmOp>`) is accurately mapped to the corresponding LLVM string (like `"llvm.riscv.buddy.bosc.maddu.mm"`) using the templates defined in the previous Pattern, covering all variants including configuration instructions, matrix multiplication instructions, element-wise operation instructions, data transfer instructions, and load/store instructions.

- Pass registration definition

​	The final `LegalizeBOSCAMEForLLVMExport` class and the `configureBOSCAMELegalizeForExportTarget` function in the file define the **rules** the compiler follows when performing this step.

1. `target.addLegalDialect<LLVM::LLVMDialect, ...>()`: declares that low-level dialects like LLVM, Arith, MemRef, etc. are 'legal'.
2. `target.addIllegalDialect<buddy::boscame::BOSCAMEDialect>()`: declares that the BOSCAME dialect is 'illegal' at the end of this Pass. This means that if any conversion rules are missing and even a single `bosc_ame` instruction remains in the code, the compiler will immediately throw an error to ensure the output is clean LLVM-level code.

​	Translate all math operations and memory operations based on Tile/Acc registers into C-style function calls that the LLVM backend can directly compile into RISC-V assembly code. Subsequently, configure the CMakeLists file and register the relevant Pass in buddy-opt under MLIR, with the Pass named `lower-bosc-ame`. The underlying support is now fully working up to this point.

- CMakeLists

```cmake
add_mlir_library(BuddyBOSCAMETransforms
  LegalizeForLLVMExport.cpp

  DEPENDS
  MLIRBOSCAMEIncGen

  LINK_LIBS PUBLIC
  BuddyBOSCAME
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRTransforms
)
```

​	Create a static or dynamic library called `BuddyBOSCAMETransforms`. Make sure that `MLIRBOSCAMEIncGen` has completed before compiling `LegalizeForLLVMExport.cpp` by following a strict build order, and inject dependencies by linking `BuddyBOSCAME`, `MLIR...Dialect` and `MLIRLLVMCommonConversion` to complete the whole build process.

```markdown
midend/
├── lib/Dialect/BOSCAME/
│   ├── CMakeLists.txt                 # Top-level build configuration (responsible for importing IR and Transforms subdirectories)
│   ├── IR/
│   │   ├── BOSCAMEDialect.cpp         # C++ specific implementation and registration logic for dialects and commands (assembly workshop)
│   │   └── CMakeLists.txt             # Packaging script for compiling the dialect ontology library (BuddyBOSCAME)
│   └── Transforms/
│       ├── LegalizeForLLVMExport.cpp  # The engine that downgrades to LLVM IR and implements rewrite logic (translator)
│       └── CMakeLists.txt             # Packaging script for the compile downgrade conversion library (BuddyBOSCAMETransforms).
```

## 4. Set up CMakeLists.txt

The main positions that need setting up are:

### 4.1 TableGen configuration

**File**: `midend/include/Dialect/BOSCAME/CMakeLists.txt` (described in 1.5)

```cmake
add_mlir_dialect(BOSCAME bosc_ame)
add_mlir_doc(BOSCAME BOSCAME Dialects/ -gen-dialect-doc)
```

### 4.2 Parent directory registers subdirectory

Add BOSCAME dialect dictionary registration and introduce the BOSCAME module:

1. **File**: `midend/include/Dialect/CMakeLists.txt`

```cmake
add_subdirectory(BOSCAME)
```

2. **File**: `midend/lib/Dialect/CMakeLists.txt`

```cmake
add_subdirectory(BOSCAME)
```

### 4.3 Global integration and initialisation

**File**: `midend/lib/CMakeLists.txt`

​	Static linking: package all the scattered dialects, conversion rules and optimisation algorithms from the entire project into a single variable called `LinkedLibs`, inject all capabilities with one click, compile the `.cpp` into `.a` or `.so` library files on the physical level, and link them to the main programme.

```cmake
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LinkedLibs
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  ${extension_libs}

  ...
  BuddyBOSCAME
  BuddyBOSCAMETransforms
  ...
)
```

**File**: `midend/lib/InitAll.cpp`

​	Dynamic registration: part of the global registration process, it's the control point where the entire BOSCAME dialect actually connects to the main compiler (like `buddy-opt`). At the **logical level** (when the C++ code runs), it calls the `BuddyBOSCAME` library from `CMakeLists.txt` to complete instantiation.

```c++
...
#include "Dialect/BOSCAME/BOSCAMEDialect.h"
...
void mlir::buddy::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<::buddy::boscame::BOSCAMEDialect>();
  ...
}
```

### 4.4 Register a dialect in the tool

**File**: `tools/buddy-opt/buddy-opt.cpp`

```c++
...
#include "BOSCAME/BOSCAMEDialect.h"
#include "BOSCAME/BOSCAMEOps.h"
#include "BOSCAME/Transform.h"
...
namespace mlir {
namespace buddy {
...
void registerLowerBOSCAMEPass();
...
} // namespace buddy
} // namespace mlir

int main(int argc, char **argv) {
  // Register all MLIR passes.
  mlir::registerAllPasses();
  ...
  mlir::buddy::registerLowerBOSCAMEPass();
  ...

  mlir::DialectRegistry registry;
  // Register all MLIR core dialects.
  registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  // Register dialects in buddy-mlir project.
  // clang-format off
  registry.insert<...
                  buddy::boscame::BOSCAMEDialect,
                  ...>();
  // clang-format on

  mlir::buddy::registerBuddyGPUTransformOps(registry);

  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "buddy-mlir optimizer driver", registry));
}
```

### 4.5 Link the library in the tool's CMakeLists.txt

**File**: `tools/buddy-opt/CMakeLists.txt`

```cmake
target_link_libraries(buddy-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIROptLib
  ...
  BuddyBOSCAME
  BuddyBOSCAMETransforms
  ...
  )
```

```markdown
tools/buddy-opt/
├── CMakeLists.txt # Link the dialect library
└── buddy-opt.cpp # Register dialect to the registry
```

## 5 Examples Instances

### 5.1 Test Demo

Write a test demo using some of the operations commands from the BOSCAME extension.

```
// RUN: buddy-opt %s --lower-bosc-ame | FileCheck %s

// ===========================================================================
// Complete Matrix Multiplication Demo using RISC-V Matrix Extension (BOSC AME)
// ===========================================================================
//
// This demo shows the complete flow of matrix multiplication:
// 1. Configure element type (msettypei)
// 2. Configure tile dimensions (msettilemi, msettileni, msettileki)
// 3. Zero accumulator (msub)
// 4. Load matrix tiles (mlae32.m, mlbe32.m)
// 5. Execute matrix multiply (mma.w.mm)
// 6. Store result (msce32.m)
//
// Matrix dimensions: C[M×N] = A[M×K] × B[K×N]
// Tile dimensions are configured via msettilem/msettilen/msettilek
//
// ===========================================================================

module {

  func.func private @print_C(i32, i32, i32, i32)
  // Demo: int32 tile-based matrix multiplication
  // Uses tile register operations (hardware-level abstraction)
  func.func @main() -> i32 {

    %c_ptr = memref.alloc() : memref<4x4xi32>     // result matrix C
    %a_ptr = memref.alloc() : memref<4x4xi32>     // matrix A
    %b_ptr = memref.alloc() : memref<4x4xi32>     // matrix B

    %stride_a = arith.constant 16 : i64           // row stride for A
    %stride_b = arith.constant 16 : i64           // row stride for B
    %stride_c = arith.constant 16 : i64           // row stride for C

    //index
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i3 = arith.constant 3 : index

    //initialize A and B with some values (for testing)
    %v0 = arith.constant 0 : i32
    %v1 = arith.constant 1 : i32
    %v2 = arith.constant 2 : i32
    %v3 = arith.constant 3 : i32
    %v4 = arith.constant 4 : i32
    %v5 = arith.constant 5 : i32
    %v6 = arith.constant 6 : i32
    %v7 = arith.constant 7 : i32
    %v8 = arith.constant 8 : i32
    %v9 = arith.constant 9 : i32
    %v10 = arith.constant 10 : i32
    %v11 = arith.constant 11 : i32
    %v12 = arith.constant 12 : i32
    %v13 = arith.constant 13 : i32
    %v14 = arith.constant 14 : i32
    %v15 = arith.constant 15 : i32
    %v16 = arith.constant 16 : i32

    memref.store %v1, %a_ptr[%i0, %i0] : memref<4x4xi32>
    memref.store %v2, %a_ptr[%i0, %i1] : memref<4x4xi32>
    memref.store %v3, %a_ptr[%i0, %i2] : memref<4x4xi32>
    memref.store %v4, %a_ptr[%i0, %i3] : memref<4x4xi32>
    memref.store %v5, %a_ptr[%i1, %i0] : memref<4x4xi32>
    memref.store %v6, %a_ptr[%i1, %i1] : memref<4x4xi32>
    memref.store %v7, %a_ptr[%i1, %i2] : memref<4x4xi32>
    memref.store %v8, %a_ptr[%i1, %i3] : memref<4x4xi32>
    memref.store %v9, %a_ptr[%i2, %i0] : memref<4x4xi32>
    memref.store %v10, %a_ptr[%i2, %i1] : memref<4x4xi32>
    memref.store %v11, %a_ptr[%i2, %i2] : memref<4x4xi32>
    memref.store %v12, %a_ptr[%i2, %i3] : memref<4x4xi32>
    memref.store %v13, %a_ptr[%i3, %i0] : memref<4x4xi32>
    memref.store %v14, %a_ptr[%i3, %i1] : memref<4x4xi32>
    memref.store %v15, %a_ptr[%i3, %i2] : memref<4x4xi32>
    memref.store %v16, %a_ptr[%i3, %i3] : memref<4x4xi32>

    memref.store %v1, %b_ptr[%i0, %i0] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i0, %i1] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i0, %i2] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i0, %i3] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i1, %i0] : memref<4x4xi32>
    memref.store %v1, %b_ptr[%i1, %i1] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i1, %i2] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i1, %i3] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i2, %i0] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i2, %i1] : memref<4x4xi32>
    memref.store %v1, %b_ptr[%i2, %i2] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i2, %i3] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i3, %i0] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i3, %i1] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i3, %i2] : memref<4x4xi32>
    memref.store %v1, %b_ptr[%i3, %i3] : memref<4x4xi32>

    // Step 1: Configure tile dimensions
    // For a simple 4x4 tile operation
    %rd = bosc_ame.msettypei 32 : i64            // msettype(e32)

    %rd_m = bosc_ame.msettilemi 4 : i64          // mtilem = 4 (rows of C and A)
    %rd_n = bosc_ame.msettileni 4 : i64          // mtilen = 4 (cols of C and B)
    %rd_k = bosc_ame.msettileki 4 : i64          // mtilek = 4 (cols of A, rows of B)

    // Step 2: Zero the accumulation register (tile register 0)
    bosc_ame.msub.w.mm 0, 0, 0

    // Step 3: Load matrix A to tile register 0 (shape: mtilem x mtilek = 4x4)
    bosc_ame.mlae32.m 0, %a_ptr, %stride_a : memref<4x4xi32>

    // Step 4: Load matrix B to tile register 1 (shape: mtilek x mtilen = 4x4)
    bosc_ame.mlbe32.m 1, %b_ptr, %stride_b : memref<4x4xi32>

    // Step 5: Execute matrix multiply: acc0 = acc0 + tile0 x tile1
    bosc_ame.mma.w.mm 0, 0, 1

    // Step 6: Store result from accumulator 0 to memory
    bosc_ame.msce32.m 0, %c_ptr, %stride_c : memref<4x4xi32>

    //row 0
    %val_c00 = memref.load %c_ptr[%i0, %i0] : memref<4x4xi32>
    %val_c01 = memref.load %c_ptr[%i0, %i1] : memref<4x4xi32>
    %val_c02 = memref.load %c_ptr[%i0, %i2] : memref<4x4xi32>
    %val_c03 = memref.load %c_ptr[%i0, %i3] : memref<4x4xi32>
    call @print_C(%val_c00, %val_c01, %val_c02, %val_c03) : (i32, i32, i32, i32) -> ()

    //row 1
    %val_c10 = memref.load %c_ptr[%i1, %i0] : memref<4x4xi32>
    %val_c11 = memref.load %c_ptr[%i1, %i1] : memref<4x4xi32>
    %val_c12 = memref.load %c_ptr[%i1, %i2] : memref<4x4xi32>
    %val_c13 = memref.load %c_ptr[%i1, %i3] : memref<4x4xi32>
    call @print_C(%val_c10, %val_c11, %val_c12, %val_c13) : (i32, i32, i32, i32) -> ()

    //row 2
    %val_c20 = memref.load %c_ptr[%i2, %i0] : memref<4x4xi32>
    %val_c21 = memref.load %c_ptr[%i2, %i1] : memref<4x4xi32>
    %val_c22 = memref.load %c_ptr[%i2, %i2] : memref<4x4xi32>
    %val_c23 = memref.load %c_ptr[%i2, %i3] : memref<4x4xi32>
    call @print_C(%val_c20, %val_c21, %val_c22, %val_c23) : (i32, i32, i32, i32) -> ()

    //row 3
    %val_c30 = memref.load %c_ptr[%i3, %i0] : memref<4x4xi32>
    %val_c31 = memref.load %c_ptr[%i3, %i1] : memref<4x4xi32>
    %val_c32 = memref.load %c_ptr[%i3, %i2] : memref<4x4xi32>
    %val_c33 = memref.load %c_ptr[%i3, %i3] : memref<4x4xi32>
    call @print_C(%val_c30, %val_c31, %val_c32, %val_c33) : (i32, i32, i32, i32) -> ()

    memref.dealloc %c_ptr : memref<4x4xi32>
    memref.dealloc %a_ptr : memref<4x4xi32>
    memref.dealloc %b_ptr : memref<4x4xi32>

    %ret = arith.constant 0 : i32
    return %ret : i32
  }

  // NOTE: High-level mma.w.mm operation (memref abstraction) requires
  // additional lowering pass to convert memref to tile operations.
  // For now, we only test the tile-level operations which map directly
  // to LLVM intrinsics.
}

// Expected lowering for tile-based operations:
// CHECK-LABEL: func.func @main
// CHECK: llvm.call @llvm.riscv.buddy.bosc.msettypei
// CHECK: llvm.call @llvm.riscv.buddy.bosc.msettilemi
// CHECK: llvm.call @llvm.riscv.buddy.bosc.msettileni
// CHECK: llvm.call @llvm.riscv.buddy.bosc.msettileki
// CHECK: llvm.call @llvm.riscv.buddy.bosc.msub.w.mm
// CHECK: llvm.call @llvm.riscv.buddy.bosc.mlae32.m
// CHECK: llvm.call @llvm.riscv.buddy.bosc.mlbe32.m
// CHECK: llvm.call @llvm.riscv.buddy.bosc.mma.w.mm
// CHECK: llvm.call @llvm.riscv.buddy.bosc.msce32.m

```

### 5.2 Makefile compilation setup

​	Add a makefile to link various command line tools together, really driving the code from high-level to low-level implementation, and finally producing assembly code that can be tested on the specified hardware.

​	Core toolchain bindings, with **`BUDDY_OPT` (`buddy-opt`)**: responsible for reading `.mlir` files and running various passes at the intermediate representation level (like `--lower-bosc-ame`); **`BUDDY_TRANSLATE` (`buddy-translate`)**: cross-domain translation, responsible for converting the `llvm` dialect in MLIR into standard LLVM IR (`.ll` files); **`BUDDY_LLC` (`llc`)**: LLVM's static compiler backend, responsible for reading LLVM IR, performing register allocation and instruction selection, and generating the final assembly file (`.s`).

​	These three Targets are for testing whether the BOSCAME matrix instructions (like `bosc_ame.mma.mm`) can correctly go through the backend. **`bosc-mma-lower`**: only performs one step, calling `--lower-bosc-ame`, to check if the translator can legalise high-level instructions into `llvm.call`; **`bosc-mma-translate`**: runs some lowering pipelines, from control flow (SCF/CF) to maths operations (Math/Arith), then to functions and memory (Func/MemRef), all lowered to the LLVM dialect, finally exporting as a `bosc-mma.ll` file; **`bosc-mma-asm`**: feeds the generated `.ll` file to the LLVM backend's `llc`, note this parameter: `-mtriple=riscv64 -mattr=+m,+v,+buddyext`, here **`+buddyext`** corresponds to the main architecture switch I defined in `RISCVBuddyExt.td`, only with this switch will LLVM generate machine assembly instructions like `mma.mm acc0, tr1, tr2`.

```cmake
#!/usr/bin/env bash

BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
BUDDY_LLC := ../../llvm/build/bin/llc

.PHONY: all clean help \
	bosc-mma bosc-mma-lower bosc-mma-translate bosc-mma-asm

all: bosc-mma

#===----------------------------------------------------------------------===#
# bosc-mma (BOSCAME MMA demo: config + load + compute + store)
#===----------------------------------------------------------------------===#

bosc-mma: bosc-mma-lower

bosc-mma-lower:
	@echo "=== Lowering BOSCAME MMA Demo ==="
	${BUDDY_OPT} bosc-mma.mlir \
	        --lower-bosc-ame \
	        -o bosc-mma-lowered.mlir
	@echo "Lowered MLIR saved to bosc-mma-lowered.mlir"

bosc-mma-translate:
	@echo "=== Translating BOSCAME MMA demo to LLVM IR ==="
	@${BUDDY_OPT} bosc-mma.mlir \
	        --lower-bosc-ame \
	        -convert-linalg-to-loops \
	        -lower-affine \
	        -convert-scf-to-cf \
	        -convert-cf-to-llvm \
	        -convert-arith-to-llvm \
	        -convert-math-to-llvm \
	        -convert-func-to-llvm \
	        -finalize-memref-to-llvm \
	        -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir \
	        -o bosc-mma.ll
	@echo "LLVM IR saved to bosc-mma.ll"

bosc-mma-asm:
	@echo "=== Generating assembly for BOSCAME MMA demo ==="
	@${BUDDY_OPT} bosc-mma.mlir \
	        --lower-bosc-ame \
	        -convert-linalg-to-loops \
	        -lower-affine \
	        -convert-scf-to-cf \
	        -convert-cf-to-llvm \
	        -convert-arith-to-llvm \
	        -convert-math-to-llvm \
	        -convert-func-to-llvm \
	        -finalize-memref-to-llvm \
	        -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
	        -mattr=+m,+v,+buddyext \
	        -o bosc-mma.s
	@echo "Assembly saved to bosc-mma.s"

clean:
	rm -f *.mlir.out *.ll *.s *-lowered.mlir
```
