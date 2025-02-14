[TOC]

## Vector Loads and Stores Instructions

| MLIR Operation       | Generated RVV Instruction | Remarks                                         |
| -------------------- | ------------------------- | ----------------------------------------------- |
| `vector.load`        | `vle<eew>`                | Supports unit stride vector load                |
| `vector.maskedload`  | Masked `vle<eew>`         | Supports unit stride vector load with mask      |
| `vector.gather`      | `vluxei<eew>`             | Indexed load, but `vloxei<eew>` is unsupported  |
| `vector.store`       | `vse<eew>`                | Supports unit stride vector store               |
| `vector.maskedstore` | Masked `vse<eew>`         | Supports unit stride vector store with mask     |
| `vector.scatter`     | `vsoxei<eew>`             | Indexed store, but `vsuxei<eew>` is unsupported |

- `vector.scatter` -> `llvm.masked.scatter` -> `vsoxei`

>  Scatter with overlapping addresses is guaranteed to be ordered from least-significant to most-significant element.

- `vector.gather` -> `llvm.masked.gather`  -> `vluxei<eew>`



### Unsupported Instructions

- **Masked Register Load:**
  - `vlm`
  - `vsm`
- **Vector Strided:**
  - `vlse<eew>`
  - `vsse<eew>`
- **Unit-stride Fault-Only-First Loads:**
  - `vle<eew>ff`
- **Vector Load/Store Segment Instructions:**
  - `vlseg<nf>e<eew>`
  - `vlsseg<nf>e<eew>`
  - `vluxseg<nf>ei<eew>`
  - `vloxseg<nf>ei<eew>`
  - `vsseg<nf>e<eew>`
  - `vssseg<nf>e<eew>`
  - `vsuxseg<nf>ei<eew>`
  - `vsoxseg<nf>ei<eew>`
- **Indexed Load/Store:**
  - `vsuxei<eew>`
  - `vloxei<eew>`
- **Vector Load/Store Whole Register:**
  - `vl<nf>re<eew>`
  - `vs<nf>r`



## Vector Reduction Instructions

- `vector.reduction` with `vector.mask` supports masked reduction instructions:
  - `vector.reduction`  -> `llvm.vector.reduce`
  
  - `vector.reduction` + `vector.mask`  ->  `llvm.vp.reduce`
  
- By setting the attribute `fastmath<reassoc>`, `vector.reduction` can generate `vfredusum` and `vfredosum` instructions.



### Unsupported Instructions

- **Vector Widening Integer Reduction:**
  - `vwredsumu`
  - `vwredsum`
- **Vector Widening Floating-Point Reduction:**
  - `vfwredusum`
  - `vfwredosum`



##  Vector Arithmetic Instructions

- `vector.mask` implements the `MaskingOpInterface` to predicate another operation:

  > The `vector.mask` is a `MaskingOpInterface` operation that predicates the execution of another operation. It takes an `i1` vector mask and an optional passthru vector as arguments.
  >
  > A implicitly `vector.yield`-terminated region encloses the operation to be masked. Values used within the region are captured from above. Only one *maskable* operation can be masked with a `vector.mask` operation at a time. An operation is *maskable* if it implements the `MaskableOpInterface`. The terminator yields all results of the maskable operation to the result of this operation.

- However, the following code doesn't lower to a valid implementation:

  ```
  %0 = vector.mask %mask, %passthru { arith.divsi %a, %b : vector<8xi32> } : vector<8xi1> -> vector<8xi32>
  ```



- All vector arithmetic instructions support only `.vv` format; widening or narrowing operations are unsupported. For example, the `addi` operation:

  > The `addi` operation takes two operands and returns one result, each of these is required to be the same type.



- `vector.fma` implements fused multiply-add, with parameters restricted to floating-point values. The conversion flow is:：`vector.fma` → `llvm.fmuladd` → `vfmadd.vv`



### Vector Integer Arithmetic Instructions

- **Vector Single-Width Integer Add and Subtract:**
  - `vadd`
  - `vsub`
  - `vrsub`
- **Vector Integer Extension:**
  - `vzext`
  - `vsext`
- **Vector Integer Add-with-Carry / Subtract-with-Borrow:**
  - `vadc`
  - `vmadc`
  - `vsbc`
  - `vmsbc`
- **Vector Bitwise Logical Instructions:**
  - `vand`
  - `vor`
  - `vxor`
- **Vector Single-Width Shift Instructions:**
  - `vsll`
  - `vsrl`
  - `vsra`
- **Vector Integer Compare Instructions:**
  - `vmseq`
  - `vmsne`
  - `vmsltu`
  - `vmslt`
  - `vmsleu`
  - `vmsle`
  - `vmsgtu`
  - `vmsgt`
- **Vector Integer Min/Max Instructions:**
  - `vminu`
  - `vmin`
  - `vmaxu`
  - `vmax`
- **Vector Single-Width Integer Multiply Instructions:**
  - `vmulhu`
  - `vmul`
  - `vmulhsu`
  - `vmulh`
- **Vector Integer Divide Instructions:**
  - `vdivu`
  - `vdiv`
  - `vremu`
  - `vrem`
- **Vector Single-Width Integer Multiply-Add Instructions:**
  - `vmacc`
  - `vnmsac`
  - `vmadd`
- **Vector Narrowing Instructions:**
  - `vnmsub`
  - `vnsrl`
  - `vnsra`
- **Vector Widening Instructions:**
  - `vwaddu`
  - `vwadd`
  - `vwsubu`
  - `vwsub`
  - `vwmulu`
  - `vwmulsu`
  - `vwmul`
  - `vwmaccu`
  - `vwmacc`



### Vector Fixed-Point Arithmetic Instructions

- **Vector Single-Width Saturating Add and Subtract:**
  - `vsadd`
  - `vssubu`
  - `vssub`
  - `vsaddu`
- **Vector Single-Width Averaging Add and Subtract:**
  - `vaaddu`
  - `vaadd`
  - `vasubu`
  - `vasub`
- **Vector Single-Width Fractional Multiply with Rounding and Saturation:**
  - `vsmul`

- **Vector Single-Width Scaling Shift Instructions:**
  - `vssrl`
  - `vssra`

- **Vector Narrowing Fixed-Point Clip Instructions:**
  - `vnclipu`
  - `vnclip`



### Vector Floating-Point Instructions

- **Vector Single-Width Floating-Point Add/Subtract:**
  - `vfadd`
  - `vfsub`
  - `vfrsub`
- **Vector Single-Width Floating-Point Multiply/Divide:**
  - `vfdiv`
  - `vfrdiv`
  - `vfmul`
- **Vector Single-Width Floating-Point Fused Multiply-Add:**
  - `vfmadd`
  - `vfnmadd`
  - `vfmsub`
  - `vfnmsub`
  - `vfmacc`
  - `vfnmacc`
  - `vfmsac`
  - `vfnmsac`
- **Vector Floating-Point Square-Root:**
  - `vfsqrt`
- **Vector Floating-Point Reciprocal Square-Root Estimate:**
  - `vfrsqrt7`
- **Vector Floating-Point Reciprocal Estimate:**
  - `vfrec7`
- **Vector Floating-Point Min/Max:**
  - `vfmin`
  - `vfmax`
- **Vector Floating-Point Sign-Injection:**
  - `vfsgnj`
  - `vfsgnjn`
  - `vfsgnjx`
- **Vector Floating-Point Compare Instructions:**
  - `vmfeq`
  - `vmfle`
  - `vmflt`
  - `vmfne`
  - `vmfgt`
  - `vmfge`
- **Vector Floating-Point Classify:**
  - `vfclass`
- **Vector Widening Instructions:**
  - `vfwadd`
  - `vfwsub`
  - `vfwmul`
  - `vfwmacc`
  - `vfwnmacc`
- **Single-Width Floating-Point/Integer Type-Convert:**
  - `vfcvt`
- **Widening Floating-Point/Integer Type-Convert:**
  - `vfwcvt`
- **Narrowing Floating-Point/Integer Type-Convert:**
  - `vfncvt`



## Vector Mask Instructions

- `vector.step` vs RVV `vid` instructions :

  |                  | `vector.step`              | `vid.v`                          |
  | ---------------- | -------------------------- | -------------------------------- |
  | **Function**     | Generate a linear sequence | Generate a linear index sequence |
  | **Range**        | `[0, N-1]`                 | `[0, vl-1]`                      |
  | **Mask Support** | Not supported              | Supported                        |

  > In `LowerVectorStep.cpp`, `arith.constant` replaced the `vector.step` operation, so it's untested whether `vector.step` can generate `vid.v` instructions.



### Unsupported Instructions

- **Vector Element Index Instruction:**
  - `vid`
- **Vector Mask-Register Logical Instructions:**
  - `vmandnot`
  - `vmand`
  - `vmor`
  - `vmxor`
  - `vmornot`
  - `vmnand`
  - `vmnor`
  - `vmxnor`
- **Vector Count Population in Mask:**
  - `vcpop.m`
- **Find-First-Set Mask Bit:**
  - `vfirst`
- **Set-Before-First Mask Bit:**
  - `vmsbf`
- **Set-Including-First Mask Bit:**
  - `vmsif`
- **Set-Only-First Mask Bit:**
  - `vmsof`
- **Vector Iota Instruction:**
  - `viota`



## Vector Permutation Instructions

- `vector.broadcast` supports broadcasting from scalar or lower-dimensional vectors to higher-dimensional vectors. `vector.splat` extends a scalar value to all elements of a result vector. Both share overlapping functionality, as they map to `vmv.v.x` and `vfmv.v.f` instructions.

  

### Unsupported Instructions

- **Scalar Move Instructions:**
  - `vmv.x.s`
  - `vfmv.f.s`
  - `vmv.s.x`
  - `vfmv.s.f`
- **Vector Slide Instructions:**
  - `vslideup`
  - `vslide1up`
  - `vfslide1up`
  - `vslidedown`
  - `vslide1down`
  - `vfslide1down`
- **Vector Register Gather Instructions:**
  - `vrgather`
  - `vrgatherei16`
- **Vector Compress Instruction:**
  - `vcompress`
- **Whole Vector Register Move:**
  - `vmv<nf>r`
- **Vector Move Instructions:**
  - `vmerge`
  - `vmv.v.v`
  - `vfmerge`
  - `vfmv.v.v`



## Configuration-Setting Instructions

The vector dialect doesn't have corresponding operations, so they cannot be directly generated.

- `vsetvli`
- `vsetivli`
- `vsetvl`
