	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl256b1p0_zvl32b1p0_zvl512b1p0_zvl64b1p0_xxiangshaname1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	kernel_attention_pv_16x1x128x17 # -- Begin function kernel_attention_pv_16x1x128x17
	.p2align	1
	.type	kernel_attention_pv_16x1x128x17,@function
kernel_attention_pv_16x1x128x17:        # @kernel_attention_pv_16x1x128x17
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -48
	.cfi_def_cfa_offset 48
	sd	s0, 40(sp)                      # 8-byte Folded Spill
	sd	s1, 32(sp)                      # 8-byte Folded Spill
	sd	s2, 24(sp)                      # 8-byte Folded Spill
	sd	s3, 16(sp)                      # 8-byte Folded Spill
	sd	s4, 8(sp)                       # 8-byte Folded Spill
	.cfi_offset s0, -8
	.cfi_offset s1, -16
	.cfi_offset s2, -24
	.cfi_offset s3, -32
	.cfi_offset s4, -40
	li	a3, 0
	li	a4, 0
	ld	a0, 136(sp)
	ld	a2, 64(sp)
	li	a5, 15
	li	a6, 127
	j	.LBB0_2
.LBB0_1:                                #   in Loop: Header=BB0_2 Depth=1
	addi	a4, a4, 1
	addi	a3, a3, 512
.LBB0_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_5 Depth 2
                                        #       Child Loop BB0_7 Depth 3
	blt	a5, a4, .LBB0_8
# %bb.3:                                #   in Loop: Header=BB0_2 Depth=1
	li	a7, 0
	mv	t0, a3
	j	.LBB0_5
.LBB0_4:                                #   in Loop: Header=BB0_5 Depth=2
	addi	a7, a7, 1
	addi	t0, t0, 512
.LBB0_5:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_7 Depth 3
	bgtz	a7, .LBB0_1
# %bb.6:                                #   in Loop: Header=BB0_5 Depth=2
	li	t1, 0
	mv	t2, t0
	bltz	a6, .LBB0_4
.LBB0_7:                                #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_5 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	add	t3, a0, t2
	sw	zero, 0(t3)
	addi	t1, t1, 1
	addi	t2, t2, 4
	bge	a6, t1, .LBB0_7
	j	.LBB0_4
.LBB0_8:
	li	a3, 0
	li	a4, 0
	li	a5, 0
	li	a6, 15
	li	a7, 112
	li	t0, 16
	li	t2, 17
	li	t1, 127
	slli	t2, t2, 9
	j	.LBB0_10
.LBB0_9:                                #   in Loop: Header=BB0_10 Depth=1
	addi	a5, a5, 1
	addi	a4, a4, 68
	add	a3, a3, t2
.LBB0_10:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_13 Depth 2
                                        #       Child Loop BB0_15 Depth 3
                                        #     Child Loop BB0_17 Depth 2
                                        #       Child Loop BB0_19 Depth 3
	blt	a6, a5, .LBB0_20
# %bb.11:                               #   in Loop: Header=BB0_10 Depth=1
	li	t3, 0
	slli	t4, a5, 7
	mv	t5, a3
	j	.LBB0_13
.LBB0_12:                               #   in Loop: Header=BB0_13 Depth=2
	addi	t3, t3, 16
	addi	t5, t5, 64
	vse32.v	v8, (t6)
.LBB0_13:                               #   Parent Loop BB0_10 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_15 Depth 3
	blt	a7, t3, .LBB0_17
# %bb.14:                               #   in Loop: Header=BB0_13 Depth=2
	li	s0, 0
	add	t6, t4, t3
	slli	t6, t6, 2
	add	t6, t6, a0
	vsetivli	zero, 16, e32, m1, ta, ma
	vle32.v	v8, (t6)
	mv	s1, t5
	mv	s2, a4
	bltz	t0, .LBB0_12
.LBB0_15:                               #   Parent Loop BB0_10 Depth=1
                                        #     Parent Loop BB0_13 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	add	s3, a1, s2
	flw	fa5, 0(s3)
	add	s3, a2, s1
	vle32.v	v9, (s3)
	vfmacc.vf	v8, fa5, v9
	addi	s0, s0, 1
	addi	s2, s2, 4
	addi	s1, s1, 512
	bge	t0, s0, .LBB0_15
	j	.LBB0_12
.LBB0_16:                               #   in Loop: Header=BB0_17 Depth=2
	fsw	fa5, 0(s0)
	addi	t3, t3, 1
	addi	t5, t5, 4
.LBB0_17:                               #   Parent Loop BB0_10 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_19 Depth 3
	blt	t1, t3, .LBB0_9
# %bb.18:                               #   in Loop: Header=BB0_17 Depth=2
	li	t6, 0
	add	s0, t4, t3
	slli	s0, s0, 2
	add	s0, s0, a0
	flw	fa5, 0(s0)
	mv	s1, t5
	mv	s2, a4
	bltz	t0, .LBB0_16
.LBB0_19:                               #   Parent Loop BB0_10 Depth=1
                                        #     Parent Loop BB0_17 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	add	s3, a2, s1
	add	s4, a1, s2
	flw	fa4, 0(s3)
	flw	fa3, 0(s4)
	fmul.s	fa4, fa3, fa4
	fadd.s	fa5, fa4, fa5
	addi	t6, t6, 1
	addi	s2, s2, 4
	addi	s1, s1, 512
	bge	t0, t6, .LBB0_19
	j	.LBB0_16
.LBB0_20:
	ld	s0, 40(sp)                      # 8-byte Folded Reload
	ld	s1, 32(sp)                      # 8-byte Folded Reload
	ld	s2, 24(sp)                      # 8-byte Folded Reload
	ld	s3, 16(sp)                      # 8-byte Folded Reload
	ld	s4, 8(sp)                       # 8-byte Folded Reload
	.cfi_restore s0
	.cfi_restore s1
	.cfi_restore s2
	.cfi_restore s3
	.cfi_restore s4
	addi	sp, sp, 48
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end0:
	.size	kernel_attention_pv_16x1x128x17, .Lfunc_end0-kernel_attention_pv_16x1x128x17
	.cfi_endproc
                                        # -- End function
	.globl	_mlir_ciface_kernel_attention_pv_16x1x128x17 # -- Begin function _mlir_ciface_kernel_attention_pv_16x1x128x17
	.p2align	1
	.type	_mlir_ciface_kernel_attention_pv_16x1x128x17,@function
_mlir_ciface_kernel_attention_pv_16x1x128x17: # @_mlir_ciface_kernel_attention_pv_16x1x128x17
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -256
	.cfi_def_cfa_offset 256
	sd	ra, 248(sp)                     # 8-byte Folded Spill
	sd	s0, 240(sp)                     # 8-byte Folded Spill
	sd	s1, 232(sp)                     # 8-byte Folded Spill
	sd	s2, 224(sp)                     # 8-byte Folded Spill
	sd	s3, 216(sp)                     # 8-byte Folded Spill
	sd	s4, 208(sp)                     # 8-byte Folded Spill
	sd	s5, 200(sp)                     # 8-byte Folded Spill
	sd	s6, 192(sp)                     # 8-byte Folded Spill
	sd	s7, 184(sp)                     # 8-byte Folded Spill
	sd	s8, 176(sp)                     # 8-byte Folded Spill
	sd	s9, 168(sp)                     # 8-byte Folded Spill
	sd	s10, 160(sp)                    # 8-byte Folded Spill
	sd	s11, 152(sp)                    # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s1, -24
	.cfi_offset s2, -32
	.cfi_offset s3, -40
	.cfi_offset s4, -48
	.cfi_offset s5, -56
	.cfi_offset s6, -64
	.cfi_offset s7, -72
	.cfi_offset s8, -80
	.cfi_offset s9, -88
	.cfi_offset s10, -96
	.cfi_offset s11, -104
	ld	t3, 64(a1)
	ld	a4, 32(a0)
	ld	a5, 40(a0)
	ld	a6, 48(a0)
	ld	a7, 56(a0)
	ld	t2, 0(a0)
	ld	t0, 8(a0)
	ld	t1, 16(a0)
	ld	a3, 24(a0)
	ld	a0, 64(a0)
	ld	t4, 0(a1)
	ld	t5, 8(a1)
	ld	t6, 16(a1)
	ld	s0, 24(a1)
	ld	s1, 32(a1)
	ld	s2, 40(a1)
	ld	s3, 48(a1)
	ld	a1, 56(a1)
	ld	s4, 64(a2)
	ld	s5, 0(a2)
	ld	s6, 8(a2)
	ld	s7, 16(a2)
	ld	s8, 24(a2)
	ld	s9, 32(a2)
	ld	s10, 40(a2)
	ld	s11, 48(a2)
	ld	ra, 56(a2)
	sd	s11, 128(sp)
	sd	s7, 96(sp)
	sd	s8, 104(sp)
	sd	s9, 112(sp)
	sd	s10, 120(sp)
	sd	a1, 64(sp)
	sd	t3, 72(sp)
	sd	s5, 80(sp)
	sd	s6, 88(sp)
	sd	s0, 32(sp)
	sd	s1, 40(sp)
	sd	s2, 48(sp)
	sd	s3, 56(sp)
	sd	a0, 0(sp)
	sd	t4, 8(sp)
	sd	t5, 16(sp)
	sd	t6, 24(sp)
	mv	a0, t2
	mv	a1, t0
	mv	a2, t1
	sd	ra, 136(sp)
	sd	s4, 144(sp)
	call	kernel_attention_pv_16x1x128x17
	ld	ra, 248(sp)                     # 8-byte Folded Reload
	ld	s0, 240(sp)                     # 8-byte Folded Reload
	ld	s1, 232(sp)                     # 8-byte Folded Reload
	ld	s2, 224(sp)                     # 8-byte Folded Reload
	ld	s3, 216(sp)                     # 8-byte Folded Reload
	ld	s4, 208(sp)                     # 8-byte Folded Reload
	ld	s5, 200(sp)                     # 8-byte Folded Reload
	ld	s6, 192(sp)                     # 8-byte Folded Reload
	ld	s7, 184(sp)                     # 8-byte Folded Reload
	ld	s8, 176(sp)                     # 8-byte Folded Reload
	ld	s9, 168(sp)                     # 8-byte Folded Reload
	ld	s10, 160(sp)                    # 8-byte Folded Reload
	ld	s11, 152(sp)                    # 8-byte Folded Reload
	.cfi_restore ra
	.cfi_restore s0
	.cfi_restore s1
	.cfi_restore s2
	.cfi_restore s3
	.cfi_restore s4
	.cfi_restore s5
	.cfi_restore s6
	.cfi_restore s7
	.cfi_restore s8
	.cfi_restore s9
	.cfi_restore s10
	.cfi_restore s11
	addi	sp, sp, 256
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end1:
	.size	_mlir_ciface_kernel_attention_pv_16x1x128x17, .Lfunc_end1-_mlir_ciface_kernel_attention_pv_16x1x128x17
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
