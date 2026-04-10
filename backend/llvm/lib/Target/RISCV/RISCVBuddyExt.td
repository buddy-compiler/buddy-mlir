//===- RISCVBuddyExt.td ---------------------------------------------------===//
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
// This is the top file for target definition of RISC-V buddy extension.
//
//===----------------------------------------------------------------------===//

include "RISCV.td"
def FeatureBuddyExt
    : SubtargetFeature<"buddyext", "HasBuddyExt", "true",
                       "'BuddyExt' (Buddy RISC-V Extension)">;
def HasBuddyExt : Predicate<"Subtarget->hasBuddyExt()">,
                            AssemblerPredicate<(all_of FeatureBuddyExt),
                            "'BuddyExt' (Buddy RISC-V Extension)">;

//===----------------------------------------------------------------------===//
// AME (RISC-V Matrix Extension) Register Definitions
//===----------------------------------------------------------------------===//
// Reference: RISC-V Matrix Extension Specification
//
// Matrix Registers:
//   - 8 Tile Registers (tr0-tr7): For input matrices A and B
//     Each tile register has MLEN bits of state
//   - 8 Accumulation Registers (acc0-acc7): For output/accumulation matrix C
//     Each accumulation register has MLEN × AMUL bits of state
//
// AMUL (Accumulation MULtiplier):
//   - Can be fractional (1/8, 1/4, 1/2) or integer (1, 2, 4, 8)
//   - Determines the width ratio between acc and tr registers
//   - For mmi8i32 (int8→int32 quad-widen), AMUL ≥ 4
//
// Data Flow:
//   Memory → tr (via mlae/mlbe) → acc (via mma/mwma/mqma) → Memory (via msce)
//===----------------------------------------------------------------------===//

let Namespace = "RISCV" in {

//===----------------------------------------------------------------------===//
// AME Tile Registers (tr0-tr7)
// Used for input matrices A and B
// Size: MLEN bits per register (hardware-defined)
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// AME Accumulation Registers (acc0-acc7)
// Used for output/accumulation matrix C
// Size: MLEN × AMUL bits per register (hardware-defined)
//
// Note: AMUL can be:
//   - Fractional (1/8, 1/4, 1/2): For C = A × Bᵀ mode with large K
//   - Integer (1, 2, 4, 8): For widening operations
//     * AMUL=4: Required for mmi8i32 (int8→int32 quad-widen)
//     * AMUL=2: Required for mmi16i32 (int16→int32 double-widen)
//     * AMUL=8: Required for mmi4i32 (int4→int32 oct-widen)
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// AME Register Classes
//===----------------------------------------------------------------------===//
// These register classes define the operand types for AME instructions
//
// Usage in instructions:
//   - TileReg: For ms1, ms2 (source operands in multiplication)
//   - AccReg: For md (destination/accumulator in multiplication)
//   - TileReg: For load/store of input matrices (A, B)
//   - AccReg: For load/store of output/accumulator (C)
//===----------------------------------------------------------------------===//

// Tile Register class (tr0-tr7)
// Used for input operands in matrix multiplication
// Note: Size is set to 256 as a placeholder; actual size depends on MLEN
def TileReg : RegisterClass<"RISCV", [untyped], 256,
                            (add TR0, TR1, TR2, TR3, TR4, TR5, TR6, TR7)> {
  let Size = 256;  // Placeholder: actual MLEN is hardware-defined
}

// Accumulation Register class (acc0-acc7)
// Used for output/accumulator in matrix multiplication
// Note: Size can be 256×AMUL where AMUL ∈ {1/8, 1/4, 1/2, 1, 2, 4, 8}
// We use 1024 as a reasonable upper bound (256 × 4 for int8→int32)
def AccReg : RegisterClass<"RISCV", [untyped], 1024,
                           (add ACC0, ACC1, ACC2, ACC3, ACC4, ACC5, ACC6, ACC7)> {
  let Size = 1024;  // Placeholder: actual MLEN×AMUL is hardware-defined
}

include "RISCVInstrInfoBuddyExt.td"
