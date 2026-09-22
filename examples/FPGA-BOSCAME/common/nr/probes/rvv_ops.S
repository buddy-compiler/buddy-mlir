/* Capability diagnostics only: deliberately exercises instructions that the
 * production NR allowlist does not yet claim to support. */
  .text
  .option push
  .option norvc
  .macro PROBE name, operation:vararg
  .balign 4
  .globl probe_\name
  .type probe_\name,@function
probe_\name:
  li t0, 16
  vsetvli t0, t0, e32, m1, ta, ma
  sd t0, 0(a3)
  flw ft0, 0(a1)
  fence rw, rw
  vle32.v v8, (a0)
  fence rw, rw
  vle32.v v9, (a1)
  fence rw, rw
  vle32.v v10, (a2)
  fence rw, rw
  \operation
  fence rw, rw
  vse32.v v10, (a2)
  fence rw, rw
  ret
  .size probe_\name, .-probe_\name
  .endm

  PROBE copy, vmv.v.v v10, v8
  PROBE add_vv, vfadd.vv v10, v8, v9
  PROBE sub_vv, vfsub.vv v10, v8, v9
  PROBE mul_vv, vfmul.vv v10, v8, v9
  PROBE div_vv, vfdiv.vv v10, v8, v9
  PROBE macc_vv, vfmacc.vv v10, v8, v9
  PROBE add_vf, vfadd.vf v10, v8, ft0
  PROBE mul_vf, vfmul.vf v10, v8, ft0
  PROBE div_vf, vfdiv.vf v10, v8, ft0
  PROBE macc_vf, vfmacc.vf v10, ft0, v8
  PROBE sqrt, vfsqrt.v v10, v9
  PROBE sum, vfredusum.vs v10, v8, v9
  PROBE ordered_sum, vfredosum.vs v10, v8, v9
  PROBE max_reduce, vfredmax.vs v10, v8, v9
  PROBE max_vv, vfmax.vv v10, v8, v9
  PROBE min_vv, vfmin.vv v10, v8, v9
  PROBE f_to_i, vfcvt.x.f.v v10, v8
  PROBE f_to_i_rtz, vfcvt.rtz.x.f.v v10, v8
  PROBE i_to_f, vfcvt.f.x.v v10, v8
  PROBE broadcast_f, vfmv.v.f v10, ft0
  PROBE zero_i, vmv.v.i v10, 0
  PROBE madd_vv, vfmadd.vv v10, v8, v9
  PROBE whole_move, vmv1r.v v10, v8

  .balign 4
  .globl probe_whole_move_vl1_e8
  .type probe_whole_move_vl1_e8,@function
probe_whole_move_vl1_e8:
  vsetivli t0, 16, e32, m1, ta, ma
  sd t0, 0(a3)
  fence rw, rw
  vle32.v v8, (a0)
  fence rw, rw
  vle32.v v10, (a2)
  fence rw, rw
  vsetivli zero, 1, e8, m1, ta, ma
  vmv1r.v v10, v8
  vsetivli zero, 16, e32, m1, ta, ma
  fence rw, rw
  vse32.v v10, (a2)
  fence rw, rw
  ret
  .size probe_whole_move_vl1_e8, .-probe_whole_move_vl1_e8

  .macro CSR_PROBE name, csr
  .balign 4
  .globl probe_\name
  .type probe_\name,@function
probe_\name:
  li t0, 16
  vsetvli t0, t0, e32, m1, ta, ma
  sd t0, 0(a3)
  li t1, 0
  csrr t1, \csr
  sw t1, 0(a2)
  ret
  .size probe_\name, .-probe_\name
  .endm
  CSR_PROBE csr_vlenb, vlenb
  CSR_PROBE csr_vl, vl
  CSR_PROBE csr_vtype, vtype

  .balign 4
  .globl probe_whole_memory
  .type probe_whole_memory,@function
probe_whole_memory:
  li t0, 16
  vsetvli t0, t0, e32, m1, ta, ma
  sd t0, 0(a3)
  fence rw, rw
  vl1re32.v v8, (a0)
  fence rw, rw
  vs1r.v v8, (a2)
  fence rw, rw
  ret
  .size probe_whole_memory, .-probe_whole_memory

  .balign 4
  .globl probe_setivli
  .type probe_setivli,@function
probe_setivli:
  li t0, 0
  vsetivli t0, 16, e32, m1, ta, ma
  sd t0, 0(a3)
  fence rw, rw
  vle32.v v8, (a0)
  fence rw, rw
  vse32.v v8, (a2)
  fence rw, rw
  ret
  .size probe_setivli, .-probe_setivli

  .macro MEMORY_PROBE name, width, lmul, count
  .balign 4
  .globl probe_\name
  .type probe_\name,@function
probe_\name:
  li t0, 0
  vsetivli t0, \count, e\width, \lmul, ta, ma
  sd t0, 0(a3)
  fence rw, rw
  vle\width\().v v8, (a0)
  fence rw, rw
  vse\width\().v v8, (a2)
  fence rw, rw
  ret
  .size probe_\name, .-probe_\name
  .endm
  MEMORY_PROBE byte_memory, 8, mf4, 16
  MEMORY_PROBE word64_memory2, 64, m1, 2
  MEMORY_PROBE word64_memory4, 64, m1, 4

  .balign 4
  .globl probe_extract_f
  .type probe_extract_f,@function
probe_extract_f:
  li t0, 16
  vsetvli t0, t0, e32, m1, ta, ma
  sd t0, 0(a3)
  fence rw, rw
  vle32.v v8, (a0)
  fence rw, rw
  vfmv.f.s ft0, v8
  fsw ft0, 0(a2)
  fence rw, rw
  ret
  .size probe_extract_f, .-probe_extract_f

  /* Test memory without requiring a vector move instruction. */
  .balign 4
  .globl probe_load_store
  .type probe_load_store,@function
probe_load_store:
  li t0, 16
  vsetvli t0, t0, e32, m1, ta, ma
  sd t0, 0(a3)
  fence rw, rw
  vle32.v v8, (a0)
  fence rw, rw
  vse32.v v8, (a2)
  fence rw, rw
  ret
  .size probe_load_store, .-probe_load_store

  /* Only illegal instructions are recoverable. RVV instructions are 32 bits,
   * so skip exactly the faulting instruction and keep the rest of the probe.
   * Every clobbered integer register is saved and restored. */
  .balign 4
  .globl nr_probe_trap
  .type nr_probe_trap,@function
nr_probe_trap:
  addi sp, sp, -32
  sd t0, 0(sp)
  sd t1, 8(sp)
  sd t2, 16(sp)
  csrr t0, mcause
  li t1, 2
  bne t0, t1, .Lfatal
  la t0, nr_probe_illegal_count
  ld t1, 0(t0)
  addi t1, t1, 1
  sd t1, 0(t0)
  csrr t1, mepc
  la t0, nr_probe_last_pc
  sd t1, 0(t0)
  csrr t2, mtval
  la t0, nr_probe_last_word
  sd t2, 0(t0)
  addi t1, t1, 4
  csrw mepc, t1
  ld t0, 0(sp)
  ld t1, 8(sp)
  ld t2, 16(sp)
  addi sp, sp, 32
  mret
.Lfatal:
  csrr a0, mcause
  csrr a1, mepc
  csrr a2, mtval
  addi sp, sp, 32
  mv a3, ra
  mv a4, sp
  tail nr_ra_trap
  .size nr_probe_trap, .-nr_probe_trap
  .option pop
  .section .note.GNU-stack,"",@progbits
