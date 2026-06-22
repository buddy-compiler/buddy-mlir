//===- RISCVInstrInfoBuddyExtBB.td - Buckyball instructions (main) -------===//
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
// Main file that includes all Buckyball Ball instruction definitions.
// This file is included by buddy-mlir's RISCVInstrInfoBuddyExt.td.
//
//===----------------------------------------------------------------------===//

// Base instructions (common to all Balls)
include "RISCVInstrInfoBuckyballBase.td"

// Ball-specific instructions
include "RISCVInstrInfoVecBall.td"
include "RISCVInstrInfoReluBall.td"
include "RISCVInstrInfoTransposeBall.td"
include "RISCVInstrInfoIm2colBall.td"
include "RISCVInstrInfoSystolicArrayBall.td"
include "RISCVInstrInfoQuantBall.td"
include "RISCVInstrInfoDequantBall.td"
