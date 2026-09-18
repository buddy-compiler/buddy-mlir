/* Capability probe only: do not whitelist these instructions until this
 * exact sequence passes on NR. Match the Triton/Buddy FP32 quantize lowering. */
  .text
  .option push
  .option norvc
  .balign 4
  .globl probe_quant
  .type probe_quant,@function
probe_quant:
  ld t0, 0(a4)
  vsetvli t0, t0, e32, m1, ta, ma
  sd t0, 0(a4)
  fence rw,rw
  vle32.v v8, (a0)
  vfabs.v v9, v8
  fence rw,rw
  vse32.v v9, (a3)
  flw ft0, 0(a2)
  vfdiv.vf v9, v8, ft0
  fmv.w.x ft0, zero
  vmfge.vf v0, v9, ft0
  li t1, 0xbf000000
  vmv.v.x v8, t1
  li t1, 0x3f000000
  vmerge.vxm v10, v8, t1, v0
  vfadd.vv v9, v9, v10
  li t1, 0xc2fe0000
  fmv.w.x ft0, t1
  vfmax.vf v9, v9, ft0
  li t1, 0x42fe0000
  fmv.w.x ft0, t1
  vfmin.vf v9, v9, ft0
  vfcvt.rtz.x.f.v v9, v9
  vsetvli zero, zero, e16, mf2, ta, ma
  vnsrl.wi v9, v9, 0
  vsetvli zero, zero, e8, mf4, ta, ma
  vnsrl.wi v9, v9, 0
  fence rw,rw
  vse8.v v9, (a1)
  fence rw,rw
  ret
  .size probe_quant, .-probe_quant
  .option pop
  .section .note.GNU-stack,"",@progbits
