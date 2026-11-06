//===- BOSCAMEMathApproxCoefficients.h - Generated f32 constants ----------===//
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
// Generated / frozen f32 polynomial and range-reduction constants for the
// authorized erf/exp approximations in BOSCAMEMathApprox.h. Values are part
// of the recipe identity; do not retune without updating contract tests.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEMATHAPPROXCOEFFICIENTS_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEMATHAPPROXCOEFFICIENTS_H

namespace buddy::boscame::mathapprox {
// clang-format off
// Range-reduction constants for exp.f32.range-reduced-taylor.v1.
inline constexpr float kLn2Hi = 0x1.6300000000000p-1f;
inline constexpr float kLn2Lo = -0x1.bd01060000000p-13f;
inline constexpr float kLog2E = 0x1.7154760000000p+0f;
// Degree-7 Taylor coefficients for the range-reduced exp remainder.
inline constexpr float kExpCoefficients[] = {
  0x1.0000000000000p+0f,
  0x1.0000000000000p+0f,
  0x1.0000000000000p-1f,
  0x1.5555560000000p-3f,
  0x1.5555560000000p-5f,
  0x1.1111120000000p-7f,
  0x1.6c16c20000000p-10f,
  0x1.a01a020000000p-13f
};
// Odd polynomial for |x| <= 0.5 in erf.f32.piecewise-taylor.v1.
inline constexpr float kErfSmallCoefficients[] = {
  0x1.20dd760000000p+0f,
  -0x1.8127460000000p-2f,
  0x1.ce2f220000000p-4f,
  -0x1.b82ce40000000p-6f,
  0x1.565bce0000000p-8f,
  -0x1.c02db40000000p-11f,
  0x1.f9a3260000000p-14f,
  -0x1.f4d25c0000000p-17f
};
// Dyadic segment centers for the large-|x| erf branches.
inline constexpr float kErfCenters[] = {
  0x1.8000000000000p-1f,
  0x1.4000000000000p+0f,
  0x1.c000000000000p+0f,
  0x1.2000000000000p+1f,
  0x1.6000000000000p+1f,
  0x1.a000000000000p+1f,
  0x1.e000000000000p+1f
};
// Degree-11 coefficients per erf segment, centered at kErfCenters[i].
inline constexpr float kErfCoefficients[7][12] = {
  {0x1.6c1c980000000p-1f, 0x1.492e420000000p-1f, -0x1.edc5640000000p-2f, 0x1.b6e85a0000000p-6f, 0x1.349b5e0000000p-3f, -0x1.b42a180000000p-5f, -0x1.b8477a0000000p-6f, 0x1.2e0afc0000000p-6f, 0x1.2db3320000000p-9f, -0x1.04105c0000000p-8f, 0x1.86234e0000000p-13f, 0x1.4725ac0000000p-11f},
  {0x1.d8865e0000000p-1f, 0x1.e465300000000p-3f, -0x1.2ebf3e0000000p-2f, 0x1.571d020000000p-3f, -0x1.93a9a80000000p-8f, -0x1.8281ce0000000p-5f, 0x1.5d00040000000p-6f, 0x1.db43ca0000000p-9f, -0x1.75672c0000000p-8f, 0x1.cc22560000000p-11f, 0x1.a007500000000p-11f, -0x1.53b1780000000p-12f},
  {0x1.f92d080000000p-1f, 0x1.b055300000000p-5f, -0x1.7a4a8a0000000p-4f, 0x1.7148c40000000p-4f, -0x1.8a0da60000000p-5f, 0x1.b222580000000p-8f, 0x1.25b3780000000p-7f, -0x1.8d10fc0000000p-8f, 0x1.7ec1d20000000p-11f, 0x1.d4cf0a0000000p-11f, -0x1.d041ee0000000p-12f, -0x1.6d70d00000000p-18f},
  {0x1.ff40480000000p-1f, 0x1.d4143a0000000p-8f, -0x1.074b600000000p-6f, 0x1.63ef620000000p-6f, -0x1.38a9840000000p-6f, 0x1.5d3b180000000p-7f, -0x1.7cae0e0000000p-9f, -0x1.5f83140000000p-11f, 0x1.0602e40000000p-10f, -0x1.8352ce0000000p-12f, -0x1.80c1340000000p-17f, 0x1.11321c0000000p-14f},
  {0x1.fff2d00000000p-1f, 0x1.3360cc0000000p-11f, -0x1.a6a51a0000000p-10f, 0x1.69cf460000000p-9f, -0x1.ab0c280000000p-9f, 0x1.6935960000000p-9f, -0x1.b2755c0000000p-10f, 0x1.52b6260000000p-11f, -0x1.7557720000000p-14f, -0x1.2abb3c0000000p-14f, 0x1.cd59240000000p-15f, -0x1.09d0a00000000p-16f},
  {0x1.ffff700000000p-1f, 0x1.e9b5e80000000p-16f, -0x1.8de3ce0000000p-14f, 0x1.9aa48a0000000p-13f, -0x1.2c7d5e0000000p-12f, 0x1.490a4e0000000p-12f, -0x1.1454640000000p-12f, 0x1.647f720000000p-13f, -0x1.56747a0000000p-14f, 0x1.b2c3200000000p-16f, -0x1.38933c0000000p-19f, -0x1.8070940000000p-19f},
  {0x1.fffffc0000000p-1f, 0x1.d9371e0000000p-21f, -0x1.bba3ac0000000p-19f, 0x1.0b6a7c0000000p-17f, -0x1.d06f580000000p-17f, 0x1.3436bc0000000p-16f, -0x1.4357b60000000p-16f, 0x1.110de40000000p-16f, -0x1.7566be0000000p-17f, 0x1.99f5c60000000p-18f, -0x1.5d69040000000p-19f, 0x1.a043120000000p-21f},
};
// clang-format on

} // namespace buddy::boscame::mathapprox

#endif
