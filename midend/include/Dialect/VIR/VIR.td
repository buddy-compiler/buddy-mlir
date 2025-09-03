//===-- VIR.td - Dynamic Vector IR Dialect Definition ------*- tablegen -*-===//
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
// This file defines the top-level structure for the Dynamic Vector IR (VIR)
// dialect, including all its components such as operations, types, and
// attributes.
//
//===----------------------------------------------------------------------===//

#ifndef VIR_TD
#define VIR_TD

include "VIR/VIRDialect.td"
include "VIR/VIROps.td"
include "VIR/VIRAttrs.td"
include "VIR/VIRTypes.td"

#endif // VIR_TD
