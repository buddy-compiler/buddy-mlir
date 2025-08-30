# Dynamic Vector Representation

For more discussion, please see [here](https://discourse.llvm.org/t/rfc-dynamic-vector-semantics-for-the-mlir-vector-dialect/75704).

## Brief Summary

This proposal extends the Vector dialect with the concept of dynamic vectors (i.e., vectors whose length may arbitrarily vary at runtime). It defines a dynamic vector type (e.g., vector<?xf32>) and two operations (vector.get_vl and vector.set_vl) to manipulate dynamic vectors.

The main focus of our proposal is to properly define the semantics of dynamic vectors. We present three generic use cases as an example of applicability but they shouldn’t prescribe or limit their usage. We also showcase RVV (RISC-V Vector Extensions) and its vector-length agnostic (VLA) model as a specific end-to-end application example. However, we envision further applicability of dynamic vectors and custom lowerings to other targets that we may explore in the future.

The dynamic vector representation seamlessly integrates with existing Vector dialect features like scalable vectors, vector masking and Linalg tiling-based vectorization.

Furthermore, we introduce an initial RVV Dialect that interfaces with the vector.get_vl and vector.set_vl operations to facilitate the lowering to RVV in LLVM. We leverage existing LLVM Vector Predication (VP) operations to model specific functionality for RVV but also reusability for future targets.

## Motivation/Context

Vector processors like some RISC-V variants can dynamically change their vector length at runtime. Yet, the current Vector dialect lacks the necessary semantics to model this dynamic behavior. Existing attempts have fallen short:

- Scalable Vector Type: This type supports VLA (Vector Length Agnostic) vectorization but can’t be used to model vector whose length may arbitrarily change at runtime.
- RVV Dialect Proposal: While supporting dynamic vector computation for RVV, this proposal remains limited to that specific architecture. Without a generic abstraction in the Vector dialect, the RVV dialect risks becoming a siloed solution, isolated from the broader vector ecosystem.

The absence of a general-purpose dynamic vector abstraction restricts the Vector dialect’s ability to express and handle dynamic vector computations effectively. Our proposal aims to fill this gap by introducing a flexible, hardware-agnostic approach to dynamic vectors within the Vector dialect itself.

## Proposal

### Dynamic Vector Type

We define a dynamic vector type as a vector whose length is defined at runtime and may arbitrarily change during the execution of the program. We use the symbol “?” to denote that the dimension of a vector is dynamic. Here are a few examples:

```
// 1-D vector type with one dynamic dimension and `i32` element type.
vector<?xi32>

// 2-D vector type with one dynamic dimension and `f32` element type.
vector<8x?xf32>
```

For simplicity, we currently limit the number of dynamic dimensions to one. We will enable dynamic dimensions in the VectorType by leveraging the existing dynamic shape infrastructure in ShapedType.

### Dynamic Vector Operations

We introduce two key operations, vector.get_vl and vector.set_vl, to manage the length of dynamic vectors:

The vector.get_vl operation retrieves the maximum vector length, in number of vector elements, that the hardware supports for a given configuration. A configuration includes a vector element type and an optional constant multiplier to scale the physical vector length of the hardware.

```
// Syntax
%vl = vector.get_vl $element_type [, $multiplier] : index

// Examples:

// Element type `i32` and length multiplier `4`.
%vl = vector.get_vl i32, 4 : index

// Element type `i32` without length multiplier.
%vl = vector.get_vl i32 : index

```

In VLA vectorization, vector.get_vl dynamically fetches the maximum vector length supported by the hardware at runtime. For instance, for a target with 128-bit maximum vector length, %vl = vector.get_vl i32, 4 : index will return 16. Likewise, for a target with 512-bit maximum vector length, it will return 64.

The vector.set_vl operation complements vector.get_vl by setting the vector length for dynamic vectors within a specific region. It takes the desired length as input, in number of vector elements, and applies it to all the dynamic vector operations within the region. vector.set_vl may also return values from the region, including dynamic vectors which will retain their vector length outside the region. Further semantics about passing dynamic vectors between vector.set_vl operations will be defined based on future real-world scenarios.

```
// Syntax
($return_value =)? vector.set_vl $vector_length : index { $op* }


// Examples: Initialize a region with a dynamic vector length specified by %vl.

// Without return value.
func.func @vector_add(%in1: memref<?xi32>, %in2: memref<?xi32>, %out: memref<?xi32>) {
  %c0 = arith.constant 0 : index  
  %dim_size = memref.dim %in1, %c0 : memref<?xi32>
  vector.set_vl %dim_size : index {
    %vec_input1 = vector.load %in1[%c0] : memref<?xi32>, vector<?xi32>
    %vec_input2 = vector.load %in2[%c0] : memref<?xi32>, vector<?xi32>
    %vec_output = arith.addi %vec_input1, %vec_input2 : vector<?xi32>
    vector.store %vec_output %out[%c0] : memref<?xi32>, vector<?xi32>
 }
}

// With return value.
func.func @vector_add(%in1: memref<?xi32>, %in2: memref<?xi32>) -> vector<?xi32> {
  %c0 = arith.constant 0 : index  
  %dim_size = memref.dim %in1, %c0 : memref<?xi32>
  %vec_ret = vector.set_vl %dim_size : index {
    %vec_input1 = vector.load %in1[%c0] : memref<?xi32>, vector<?xi32>
    %vec_input2 = vector.load %in2[%c0] : memref<?xi32>, vector<?xi32>
    %vec_output = arith.addi %vec_input1, %vec_input2 : vector<?xi32>
    return %vec_output : vector<?xi32>
  } -> vector<?xi32>
  return %vec_ret : vector<?xi32>
}

```

The flexibility of vector.get_vl and vector.set_vl allows us to dynamically adjust vector lengths for optimal performance across diverse hardware implementations with VLA support.

We intentionally keep the design open for nested vector.set_vl regions. This flexibility caters to scenarios where the vector length might need to change within the same computation, like for some bitcast operations or vectorization of control flow. While we haven’t encountered many such use cases in practice yet, we plan to revisit nested region support as relevant scenarios emerge.

### Potential Uses

This section showcases how the vector.get_vl and vector.set_vl operations for dynamic vectors can be applied in diverse scenarios, even spanning different levels of abstraction.

```
// Skeleton of driving example.

#map = affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>

func.func @vector_add(%input1: memref<?xi32>, %input2: memref<?xi32>, %output: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  // Get the dimension of the workload.
  %dim_size = memref.dim %input1, %c0 : memref<?xi32>
  // Perform dynamic vector addition.
  [ ... see use cases below ...]
}
```

**Use Case 1 - Linalg Vectorization: Set vector length per loop iteration.**

In this use case, vector.set_vl is dynamically adjusted within the loop body. For all iterations except the last, the vector length is set to the value returned by vector.get_vl. However, the final iteration may have a smaller vector length.

```
// Returns four times the physical vl for element type i32.
%vl = vector.get_vl i32, 4 : index

scf.for %idx = %c0 to %dim_size step %vl { // Tiling
  %it_vl = affine.min #map(%idx)[%vl, %dim_size]
  vector.set_vl %it_vl : index {
    %vec_input1 = vector.load %input1[%idx] : memref<?xi32>, vector<?xi32>
    %vec_input2 = vector.load %input2[%idx] : memref<?xi32>, vector<?xi32>
    %vec_output = arith.addi %vec_input1, %vec_input2 : vector<?xi32>
    vector.store %vec_output %output[%idx] : memref<?xi32>, vector<?xi32>
  }
}
```

**Use Case 2 - Linalg Vectorization: Set vector length once per main vector loop and once per epilogue loop.**

This use case demonstrates setting the vector length once at the beginning of the main vector loop for efficient bulk processing. Additionally, it sets the vector length again in the epilogue loop to handle the remaining elements that were not processed by the main vector loop.

```
// Returns four times the physical vl for element type i32.
%vl = vector.get_vl i32, 4 : index
%steps = arith.floordivsi %dim_size, %vl : index
%main_ub = arith.multi %steps, %vl : index
vector.set_vl %vl : index {
  scf.for %idx = %c0 to %main_ub step %vl {
    %vec_input1 = vector.load %input1[%idx] : memref<?xi32>, vector<?xi32>
    %vec_input2 = vector.load %input2[%idx] : memref<?xi32>, vector<?xi32>
    %vec_output = arith.addi %vec_input1, %vec_input2 : vector<?xi32>
    vector.store %vec_output %output[%idx] : memref<?xi32>, vector<?xi32>
  }
}

%rem_ub = %dim_size - %main_ub
%cond = arith.cmpi sgt, %rem_ub, %c0 : index
scf.if %cond {
  vector.set_vl %rem_ub : index {
    %vec_input1 = vector.load %input1[%rem_idx] : memref<?xi32>, vector<?xi32>
    %vec_input2 = vector.load %input2[%rem_idx] : memref<?xi32>, vector<?xi32>
    %vec_output = arith.addi %vec_input1, %vec_input2 : vector<?xi32>
    vector.store %vec_output %output[%rem_idx] : memref<?xi32>, vector<?xi32> 
  }
}
```

**Use Case 3: Set vector length to the full dimension size.**

In this case, the vector length is set to match the full dimension size of the data, applying vector operations across the full dataset. This demonstrates how vector.set_vl can effectively represent a “vector loop” at a higher level.

```
// The `dim_size` represents the whole size of the workload.
vector.set_vl %dim_size : index {
  %vec_input1 = vector.load %input1[%c0] : memref<?xi32>, vector<?xi32>
  %vec_input2 = vector.load %input2[%c0] : memref<?xi32>, vector<?xi32>
  %vec_output = arith.addi %vec_input1, %vec_input2 : vector<?xi32>
  vector.store %vec_output %output[%c0] : memref<?xi32>, vector<?xi32>
}
```

This use case is purely for illustration, as a full design and implementation poses potential challenges beyond our current scope of work.

### Initial RVV Dialect

Effectively lowering dynamic vectors to the RISC-V backend in LLVM requires bridging the gap between the generic representation and RVV-specific features in MLIR. We propose a minimalist RVV dialect to address this need which, for now, will only have the rvv.set_vl operation to facilitate the lowering of vector.get_vl and vector.set_vl. We plan to leverage the VP intrinsics defined within the LLVM dialect to represent dynamic vector computation and memory operations. We anticipate adding more specific RVV operations in the future to further enhance RVV support.

**RVV Lowering Example**

The lowering process from generic dynamic vectors to RVV involves several key steps:

- Lowering vector.get_vl/set_vl to rvv.set_vl.
- Lowering vector computations to VP operations.
- Converting dynamic vector types to scalable vector types using the LLVM convention for RVV.

Here is the lowering for the vector addition example from use case #1 above:

```
%vlmax = rvv.set_vl %max, i32, 4 : index
scf.for $idx = %c0 to %dim_size step %vlmax { // Tiling
  %it_vl = affine.min #map(%idx)[%vlmax, %dim_size]
  %vl = rvv.set_vl %it_vl, i32, 4 : index
  %mask = vector.create_mask %vl : vector<[8]xi1>
  %iptr_1 = ... ... // Resolve the input1 memref pointer
  %iptr_2 = ... ... // Resolve the input2 memref pointer
  %iptr_out = ... ... // Resolve the output memref pointer
  %vec_1 = "llvm.intr.vp.load" (%iptr_1, %mask, %vl) :
        (!llvm.ptr<i32>, vector<[8]xi1>, i32) -> vector<[8]xf32>
  %vec_2 = "llvm.intr.vp.load" (%iptr_2, %mask, %vl) :
        (!llvm.ptr<i32>, vector<[8]xi1>, i32) -> vector<[8]xf32>
  %res = "llvm.intr.vp.add" (%vec1, %vec2, %mask, %vl) :
        (vector<[8]xi32>, vector<[8]xi32>, vector<[8]xi1>, i32) -> vector<[8]xi32>
  "llvm.intr.vp.store" (%res, %iptr_out, %mask, %vl) :
        (vector<[8]xf32>, !llvm.ptr<f32>, vector<[8]xi1>, i32) -> ()
}
```
