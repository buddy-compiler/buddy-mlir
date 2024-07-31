# Llama 2 Operator/Layer level instance extraction

---

## Operator Level：

### **TOSA Dialect**

#### `tosa.mul`

	make next-mul-run

- **Input Tensors**:
  - Shape: `tensor<1xf32>`
  - Example: `[3.0]`

  - Shape: `tensor<1x40x1xf32>`
  - Example: `[[[2.0], [2.0], ..., [2.0]]]` (40 elements)
- **Output Tensor**:
  - Shape: `tensor<1x40x1xf32>`
  - Example: All elements will be `6.0` after the multiplication operation.
- **Multiplication Operation**:
  - The `tosa.mul` operation is applied to the input tensors `%arg0` and `%arg1`, performing an element-wise multiplication.
- **Timing:**
  - elapsed time: 0.000380993

#### `tosa.negate`

	make next-negate-run

- **Input Tensor**:
  - Shape: `tensor<1x32x40x64xf32>`
  - Example: All elements initialized to `1.0`.
- **Output Tensor**:
  - Shape: `tensor<1x32x40x64xf32>`
  - Example: All elements will be `-1.0` after the negate operation.
- **Negate Operation**:
  - The `tosa.negate` operation is applied to the input tensor `%arg0`, which negates each element in the tensor.
- **Timing:**
  - elapsed time: 0.000413179

#### `tosa.reciprocal`

	make next-reciprocal-run

- **Input Tensor**:
  - Shape: `tensor<1x10xf32>`
  - Example: All elements initialized to `[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]`.
- **Output Tensor**:
  - Shape: `tensor<1x10xf32>`
  - Example: All elements will be the reciprocal of the input tensor elements, i.e., `[1.0, 0.5, 0.333, 0.25, 0.2, 0.166, 0.142, 0.125, 0.111, 0.1]`.
- **Reciprocal Operation**:
  - The `tosa.reciprocal` operation is applied to the input tensor `%arg0`, which computes the reciprocal (1/x) of each element in the tensor.
- **Timing:**
  - elapsed time: 0.000286102

#### `tosa.reduce_sum`

	make next-reducesum-run

- **Input Tensor**:
  - Shape: `tensor<1x40x4096xf32>`
  - Example: All elements initialized to `1.0`.
- **Output Tensor**:
  - Shape: `tensor<1x40x1xf32>`
  - Example: Each element in the output tensor is the sum of 4096 elements from the corresponding dimension of the input tensor, which will be `4096.0` for each element.
- **Reduce Sum Operation**:
  - The `tosa.reduce_sum` operation is applied to the input tensor `%arg0`, summing elements along the `axis=2` dimension. This reduces the shape of the tensor from `[1, 40, 4096]` to `[1, 40, 1]`.
- **Timing:**
  - elapsed time: 0.000262976

#### `tosa.rsqrt`

	make next-rsqrt-run

- **Input Tensor**:
  - Shape: `tensor<1x40x1xf32>`
  - Example: All elements initialized to `3.0`.
- **Output Tensor**:
  - Shape: `tensor<1x40x1xf32>`
  - Example: Each element in the output tensor will be the reciprocal of the square root of the corresponding element in the input tensor, which will be approximately `0.57735` for each element.
- **Rsqrt Operation**:
  - The `tosa.rsqrt` operation is applied to the input tensor `%arg0`, which computes the reciprocal of the square root of each element in the tensor.
- **Timing:**
  - elapsed time: 3.09944e-06

#### `tosa.transpose`

	make next-transpose-run

- **Input Tensor**:
  - Shape: `tensor<1x40x32x128xf32>`
  - Example: All elements initialized to `1.0`.
- **Output Tensor**:
  - Shape: `tensor<1x32x40x128xf32>`
  - Example: The tensor after transposing will have the elements permuted according to the permutation vector `[0, 2, 1, 3]`. Given that all elements are initialized to `1.0`, the values remain `1.0` but the shape is permuted.
- **Transpose Operation**:
  - The `tosa.transpose` operation is applied to the input tensor `%arg0` with the permutation vector `%perm`, which rearranges the dimensions of the input tensor according to `[0, 2, 1, 3]`.- The permutation `[0, 2, 1, 3]` means:
    - The first dimension remains the same.
    - The second dimension (40) is swapped with the third dimension (32).
    - The fourth dimension (128) remains the same.
  - Therefore, the input tensor shape `[1, 40, 32, 128]` is transposed to `[1, 32, 40, 128]`.
- **Timing:**
  - elapsed time: 0.000138044

### **Math Dialect**

#### `math.fpowi`

	make next-fpowi-run

- **Input Tensor**:
  - Shape: `tensor<1x32x40x64xf32>`
    - Example: All elements initialized to `5.0`.
- **Output Tensor**:
  - Shape: `tensor<1x32x40x64xf32>`
    - Example: Each element in the output tensor will be the value of the corresponding element in the input tensor raised to the power of `2`, i.e., `25.0` for each element.
- **Power Operation**:
  - The `math.fpowi` operation is applied to each element in the input tensor `%arg0`, raising it to the power of `2`.
  - For example, if an element in the input tensor is `5.0`, the corresponding element in the output tensor will be `5.0^2 = 25.0`.
- **Timing:**
  - elapsed time: 8.29697e-05

### **Linalg Dialect**

#### `linalg.matmul`

make next-matmul-run

- **Input Tensors**:
  - Shape: `tensor<40x4096xf32>`
  - Example: All elements initialized to `3.0`.

  - Shape: `tensor<4096x4096xf32>`
  - Example: All elements initialized to `2.0`.
- **Output Tensor**:
  - Shape: `tensor<40x4096xf32>`
  - Example: Each element in the output tensor will be the result of the matrix multiplication of the input tensors. Given the initialization, the elements will be the result of `3.0 * 2.0 * 4096`.
- **Matrix Multiplication Operation**:
  - The `linalg.matmul` operation is applied to the input tensors `%arg0` and `%arg1`, performing matrix multiplication.
  - The output tensor `%arg2` is the result of the matrix multiplication, where each element is calculated as the sum of the element-wise products of the rows of the first matrix and the columns of the second matrix.
- **Timing:**
  - elapsed time: 7.42794

---

## Layer Level

#### `Full Connect Layer`

	`make next-fc-run`

- **Input Tensors**:
  - Shape: `tensor<1x40x4096xf32>`
  - Example: All elements initialized to `3.0`.

  - Shape: `tensor<4096x4096xf32>`
  - Example: All elements initialized to `2.0`.

  - Shape: `tensor<4096x4096xf32>`
  - Example: All elements initialized to `1.0`.

  - Shape: `tensor<1x40x4096xf32>`
  - Example: All elements initialized to `4.0`.
- **Output Tensor**:
  - Shape: `tensor<1x40x4096xf32>`
  - Example: The exact values will depend on the computations performed during the fully connected layer operations, which include multiplication, transposition, and reshaping.
- **Fully Connected Layer Operations**:
  1. **Multiplication**:
     - `%41 = tosa.mul %arg0, %arg3` multiplies the elements of `%arg0` and `%arg3` element-wise.
     - Example: The result tensor will have elements initialized to `3.0 * 4.0 = 12.0`.
  2. **Transpose**:
     - `%43 = tosa.transpose %arg1, %42` transposes the tensor `%arg1` according to the permutation `[1, 0]`.
     - Example: The tensor shape remains `[4096x4096]`.
  3. **Reshape**:
     - `%44 = tosa.reshape %41` reshapes the tensor from `tensor<1x40x4096xf32>` to `tensor<40x4096xf32>`.
  4. **Matrix Multiplication**:
     - `%45 = linalg.matmul` performs matrix multiplication on the reshaped tensor and the transposed tensor.
     - Example: Each element of the resulting `tensor<40x4096xf32>` will be `12.0 * 2.0 * 4096 = 98304.0`.
     - The result is reshaped back to `tensor<1x40x4096xf32>`.
  5. **Second Transpose and Reshape**:
     -  Similar transpose and reshape operations are performed on `%arg2` and the result tensor `%41`.
  6. **Second Matrix Multiplication**:
     - `%50 = linalg.matmul` performs matrix multiplication on the reshaped tensors, and the result is reshaped back to `tensor<1x40x4096xf32>`.
     - Example: Each element of the resulting `tensor<40x4096xf32>` will be `12.0 * 1.0 * 4096 = 49152.0`.
     - The final output tensor will have the shape `tensor<1x40x4096xf32>` with each element being `49152.0`.
- **Timing:**
  - elapsed time: 10.8429

#### `Feed Forward Network`

	`make next-ffn-run`

- **Input Tensors**:
  - Shape: `tensor<1x40x4096xf32>`
  - Example: All elements initialized to `3.0`.

  - Shape: `tensor<4096xf32>`
  - Example: All elements initialized to `1.0`.

  - Shape: `tensor<11008x4096xf32>`
  - Example: All elements initialized to `1.0`.

  - Shape: `tensor<11008x4096xf32>`
  - Example: All elements initialized to `2.0`.

  - Shape: `tensor<4096x11008xf32>`
  - Example: All elements initialized to `1.0`.
- **Output Tensor**:
  - Shape: `tensor<1x40x4096xf32>`
- **Feed Forward Network Operations**:
  1. **Multiplication**:
     - `%138 = tosa.reshape %arg9 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>`
     - `%139 = tosa.mul %138, %arg0 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>`
     - Example: The result tensor will have elements initialized to `1.0 * 3.0 = 3.0`.
  2. **Transpose**:
     - `%141 = tosa.transpose %arg10, %140 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>`
     - Example: The tensor shape remains `[4096x11008]`.
  3. **Reshape**:
     - `%142 = tosa.reshape %139 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>`
  4. **Matrix Multiplication**:
     - `%143 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%142, %141 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_24 : tensor<40x11008xf32>) -> tensor<40x11008xf32>`
     - Example: Each element of the resulting `tensor<40x11008xf32>` will be the sum of the products of corresponding elements from the input tensor and the transposed weight tensor, resulting in a tensor with elements calculated as `3.0 * 1.0 * 4096 = 12288.0`.
  5. **Reshape**:
     - `%144 = tosa.reshape %143 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>`
  6. **Sigmoid and Multiplication**:
     - `%145 = tosa.sigmoid %144 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>`
     - Example: Each element of the resulting tensor will be the sigmoid of `12288.0`, which is very close to `1.0` because the sigmoid function asymptotically approaches `1` for large positive inputs.
     - `%146 = tosa.mul %144, %145 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>`
     - Example: Each element of the resulting tensor will be `12288.0 * 1.0 = 12288.0`.
  7. **Second Transpose and Reshape**:
     - Similar transpose and reshape operations are performed on `%arg11` and the result tensor `%146`.
  8. **Second Matrix Multiplication**:
     - `%150 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%149, %148 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_25 : tensor<40x11008xf32>) -> tensor<40x11008xf32>`
     - Example: Each element of the resulting tensor will be `3.0 * 2.0 * 4096 = 24576.0`.
     - The result is reshaped back to `tensor<1x40x11008xf32>`.
  9. **Final Multiplication and Matrix Multiplication**:
     - `%152 = tosa.mul %146, %151 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>`
     - Example: Each element of the resulting tensor will be `12288.0 * 24576.0 = 301989888.0`.
     - `%156 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%155, %154 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_26 : tensor<40x4096xf32>) -> tensor<40x4096xf32>`
     - Example: Each element of the resulting tensor will be the sum of products of elements from the tensor of `301989888.0` and the weight tensor, resulting in very large values.
  10. **Addition**:
      - `%158 = tosa.add %arg0, %157 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>`
      - The final output tensor will be the addition of the original input tensor `%arg0` and the resulting tensor from the previous computations.
- **Timing:**
  - elapsed time: 56.0974

#### `RMSNorm`

	`make next-rmsnorm-run`

- **Input Tensor**:
  - Shape: `tensor<1x40x4096xf32>`
  - Example: All elements initialized to `3.0`.
- **Output Tensor**:
  - Shape: `tensor<1x40x4096xf32>`
  - Example: Each element in the output tensor will be the result of the RMSNorm operations applied to the input tensor.
- **RMSNorm Operations**:
  1. **Square Elements**:
     - `%31 = linalg.generic` squares each element in the input tensor `%arg0`.
     - Example: Each element will be `3.0^2 = 9.0`.
  2. **Reduce Sum**:
     - `%32 = tosa.reduce_sum %31 {axis = 2 : i32}` sums the squared elements along the last dimension.
     - Example: Each element in the resulting tensor will be the sum of `4096` squared elements, `9.0 * 4096 = 36864.0`.
  3. **Reciprocal and Multiplication**:
     - `%34 = tosa.reciprocal %33` computes the reciprocal of a constant tensor value `4096.0`.
     - `%35 = tosa.mul %34, %32` multiplies the reciprocal with the sum of squares.
     - Example: Each element will be `1/4096 * 36864.0 = 9.0`.
  4. **Add Small Constant**:
     - `%37 = tosa.add %35, %36` adds a small constant `1e-5` to the result.
     - Example: Each element will be `9.0 + 1e-5`.
  5. **Reciprocal Square Root**:
     - `%38 = tosa.rsqrt %37` computes the reciprocal square root of the result.
     - Example: Each element will be approximately `1 / sqrt(9.0 + 1e-5) ≈ 0.333333`.
  6. **Final Multiplication**:
     - `%39 = tosa.mul %arg0, %38` multiplies the original input tensor `%arg0` by the reciprocal square root.
     - Example: Each element will be `3.0 * 0.333333 = 0.999999`.
- **Timing:**
  - elapsed time: 0.000798941

#### `Softmax`

	`make next-softmax-run`

- **Input Tensors**:

  - Shape: `tensor<1x32x40x40xf32>`
  - Example: All elements initialized to `3.0`.

  - Shape: `tensor<1x1x40x40xf32>`
  - Example: All elements initialized to `0.0`.

- **Output Tensor**:

  - Shape: `tensor<1x32x40x40xf32>`
  - Example: Each element in the output tensor will be the result of the softmax operations applied to the input tensor. The elements will sum to `1` along the last axis (softmax dimension).

- **Softmax Operations**:

  1. **Scaling**:

    - `%101 = tosa.reciprocal %100` computes the reciprocal of a constant tensor value `11.3137083`.
    - `%102 = tosa.mul %arg0, %101` scales the input tensor `%arg0` by multiplying with the reciprocal.
    - Example: Each element will be `3.0 / 11.3137083 ≈ 0.265`.

  2. **Addition**:
     - `%103 = tosa.add %102, %arg1` adds the second input tensor `%arg1` to the scaled tensor.
     - Example: Each element will remain `0.265` as `%arg1` is all zeros.
  3. **Max Reduction**:
     - `%104 = tosa.reduce_max %103` computes the maximum value along the last dimension (axis 3).
     - Example: The maximum value along each `40x40` slice will be `0.265`.
  4. **Subtraction**:
     - `%105 = tosa.sub %103, %104` subtracts the maximum value from each element to ensure numerical stability.
     - Example: Each element will be `0.265 - 0.265 = 0.0`.
  5. **Exponentiation**:
     - `%106 = tosa.exp %105` applies the exponential function to each element.
     - Example: Each element will be `exp(0.0) = 1.0`.
  6. **Sum Reduction**:
     - `%107 = tosa.reduce_sum %106` computes the sum of exponentials along the last dimension (axis 3).
     - Example: The sum along each `40x40` slice will be `40` since each element is `1.0`.
  7. **Reciprocal of Sum**:
     - `%108 = tosa.reciprocal %107` computes the reciprocal of the sum of exponentials.
     - Example: Each element will be `1 / 40 = 0.025`.
  8. **Final Multiplication**:
     - `%109 = tosa.mul %106, %108` multiplies the exponentials by the reciprocal of their sum to normalize them.
     - Example: Each element will be `1.0 * 0.025 = 0.025`.

- **Timing:**

  - elapsed time: 0.000925779

#### `Self-Attention`

	`make next-selfattention-run`

- **Input Tensors**:
  - `tensor<1x1x4096xf32>`
    - Shape: `[1, 1, 4096]`
    - Example: All elements initialized to `3.0`.
  - `tensor<1x40x4096xf32>`
    - Shape: `[1, 40, 4096]`
    - Example: All elements initialized to `1.0`.
  - `tensor<40xi64>`
    - Shape: `[40]`
    - Example: All elements initialized to `2`.
  - `tensor<4096x4096xf32>`
    - Shape: `[4096, 4096]`
    - Example: All elements initialized to `1.0`.
  - `tensor<4096x4096xf32>`
    - Shape: `[4096, 4096]`
    - Example: All elements initialized to `1.0`.
  - `tensor<4096x4096xf32>`
    - Shape: `[4096, 4096]`
    - Example: All elements initialized to `1.0`.
  - `tensor<1x1x2048x128xf32>`
    - Shape: `[1, 1, 2048, 128]`
    - Example: All elements initialized to `1.0`.
  - `tensor<1x1x2048x128xf32>`
    - Shape: `[1, 1, 2048, 128]`
    - Example: All elements initialized to `1.0`.
  - `tensor<4096x4096xf32>`
    - Shape: `[4096, 4096]`
    - Example: All elements initialized to `2.0`.
  - `tensor<1x1x40x40xf32>`
    - Shape: `[1, 1, 40, 40]`
    - Example: All elements initialized to `0.0`.
- **Output Tensor**:
  - Shape: `tensor<1x40x4096xf32>`
  - Example: Each element in the output tensor will be the result of the self-attention operations applied to the input tensors.
- **Softmax Operations**:
  1. **Compute Query, Key, and Value Matrices**:
     - **Query**:
       - `%41 = tosa.mul %arg0, %arg1` scales the input tensor.
       - Example: Each element will be `3.0 * 1.0 = 3.0`.
       - `%45` and `%46` involve transposition and reshaping.
       - Example: Elements remain `3.0`.
     - **Key**:
       - `%50` and `%51` involve similar transposition and reshaping as Query.
       - Example: Elements remain `3.0`.
     - **Value**:
       - `%55` and `%56` involve similar transposition and reshaping as Query.
       - Example: Elements remain `3.0`.
  2. **Apply Rotary Positional Encoding (RoPE) to Q and K Vectors**:
     - **Query RoPE**:
       - Transpose and reshape operations (`%57`, `%58`, `%59`).
       - Example: Shape transformed to `1x32x40x128`.
     - **Key RoPE**:
       - Similar transpose and reshape operations (`%60`, `%61`, `%62`).
     - **Value RoPE**:
       - Similar transpose and reshape operations (`%63`, `%64`, `%65`).
  3. **Compute Softmax(Q, K) and Self-Attention Output**:

     - **Attention Scores**:
       - Extract slices and generic operations to calculate (`%66` to `%79`).
       - Compute multiplication of Q and K with positional encoding applied (`%80` to `%83`).
     - **Softmax**:
       - `%84 = tosa.add %80, %83` sums the attention scores.
       - Apply softmax function over the attention scores.
     - **Self-Attention Output**:
       - Compute the output by multiplying attention scores with Value matrix (`%112` to `%116`).
       - Transpose and reshape operations for final output (`%117` to `%121`).
       - Example: Each element in the output tensor will be influenced by the weighted sum of values, resulting from the softmax-scaled dot product of queries and keys.
  4. **Final Matrix Multiplication and Addition**:
     - `%125 = linalg.matmul` performs matrix multiplication on the reshaped tensor and the transposed weight tensor.
     - `%127 = tosa.add %arg1, %126` adds the original input tensor `%arg1` to the result of the matrix multiplication.
     - Example: The output tensor shape is `tensor<1x40x4096xf32>`, and each element will be the sum of the original input tensor elements and the matrix multiplication result.
- **Timing:**
  - elapsed time: 48.4356