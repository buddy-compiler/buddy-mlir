/* Adapted from ModelZoo runtime/asm/qwen35_rvv_copy.S at commit
 * 8815b74fb6d3cd6288c4d99ac6fd7c5d041a7cc3. This is the NR-validated
 * e32,m1 copy subset: at most 16 lanes per issue and fences at both sides
 * of every vector memory operation. Additional scalar paths handle bytes.
 * C ABI: void nr_copy_bytes(void *dst, const void *src, size_t bytes).
 * Source and destination must not overlap. v8/vl/vtype are caller-clobbered. */
  .text
  .balign 4
  .globl nr_copy_bytes
  .type nr_copy_bytes,@function
nr_copy_bytes:
  beqz a2, .Ldone
  or t1, a0, a1
  andi t1, t1, 3
  bnez t1, .Lbytes
  andi a3, a2, 3
  srli a2, a2, 2
  beqz a2, .Ltail
.Lwords:
  li t0, 16
  bgeu a2, t0, .Lsetvl
  mv t0, a2
.Lsetvl:
  vsetvli t0, t0, e32, m1, ta, ma
  fence rw, rw
  vle32.v v8, (a1)
  fence rw, rw
  vse32.v v8, (a0)
  fence rw, rw
  sub a2, a2, t0
  slli t0, t0, 2
  add a0, a0, t0
  add a1, a1, t0
  bnez a2, .Lwords
.Ltail:
  mv a2, a3
  beqz a2, .Ldone
.Lbytes:
  lbu t0, 0(a1)
  sb t0, 0(a0)
  addi a0, a0, 1
  addi a1, a1, 1
  addi a2, a2, -1
  bnez a2, .Lbytes
  fence rw, rw
.Ldone:
  ret
  .size nr_copy_bytes, .-nr_copy_bytes
  .section .note.GNU-stack,"",@progbits
