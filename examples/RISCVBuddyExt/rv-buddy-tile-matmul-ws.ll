@.str = private unnamed_addr constant [4 x i8] c"%d \00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@.str.2 = private unnamed_addr constant [24 x i8] c"Start slow CPU matmul.\0A\00", align 1
@.str.3 = private unnamed_addr constant [19 x i8] c"Cycles taken: %lu\0A\00", align 1
@.str.4 = private unnamed_addr constant [23 x i8] c"Start gemmini matmul.\0A\00", align 1
@.str.5 = private unnamed_addr constant [19 x i8] c"Cycles token: %lu\0A\00", align 1
@.str.6 = private unnamed_addr constant [4 x i8] c"C:\0A\00", align 1
@.str.7 = private unnamed_addr constant [7 x i8] c"Gold:\0A\00", align 1
@.str.8 = private unnamed_addr constant [24 x i8] c"tile_I is non-positive\0A\00", align 1
@.str.9 = private unnamed_addr constant [24 x i8] c"tile_J is non-positive\0A\00", align 1
@.str.10 = private unnamed_addr constant [24 x i8] c"tile_K is non-positive\0A\00", align 1
@.str.11 = private unnamed_addr constant [51 x i8] c"tile_I is too large (tile_I * DIM > dim_I_padded)\0A\00", align 1
@.str.12 = private unnamed_addr constant [51 x i8] c"tile_J is too large (tile_J * DIM > dim_J_padded)\0A\00", align 1
@.str.13 = private unnamed_addr constant [51 x i8] c"tile_K is too large (tile_K * DIM > dim_K_padded)\0A\00", align 1
@.str.14 = private unnamed_addr constant [58 x i8] c"Not enough space in scratchpad to store A and B matrices\0A\00", align 1
@.str.15 = private unnamed_addr constant [44 x i8] c"Not enough space in accumulator to store C\0A\00", align 1
@.str.16 = private unnamed_addr constant [101 x i8] c"I, J, and K tiling factors must be less than 65535, to fit within the bounds of the LOOP_WS function\00", align 1
@__const.tiled_matmul.matmul_type_str = private unnamed_addr constant [3 x [4 x i8]] [[4 x i8] c"OS\00\00", [4 x i8] c"WS\00\00", [4 x i8] c"CPU\00"], align 1
@.str.17 = private unnamed_addr constant [60 x i8] c"Not implemented: %s matmul, a_transpose=%d, b_transpose=%d\0A\00", align 1
@.str.18 = private unnamed_addr constant [49 x i8] c"Not implemented: %s matmul, full_C=%d, low_D=%d\0A\00", align 1
@.str.19 = private unnamed_addr constant [36 x i8] c"Not implemented: %s matmul, act=%d\0A\00", align 1
@.str.20 = private unnamed_addr constant [97 x i8] c"When doing layernorm or softmax, the full J dimension of the matrix must fit in the accumulator\0A\00", align 1
@.str.21 = private unnamed_addr constant [12 x i8] c"config_ex.\0A\00", align 1
@.str.22 = private unnamed_addr constant [12 x i8] c"config_st.\0A\00", align 1
@.str.23 = private unnamed_addr constant [12 x i8] c"config_ld.\0A\00", align 1
@.str.24 = private unnamed_addr constant [24 x i8] c"loop_ws_config_bounds.\0A\00", align 1
@.str.25 = private unnamed_addr constant [26 x i8] c"loop_ws_config_addrs_ab.\0A\00", align 1
@.str.26 = private unnamed_addr constant [26 x i8] c"loop_ws_config_addrs_dc.\0A\00", align 1
@.str.27 = private unnamed_addr constant [28 x i8] c"loop_ws_config_strides_ab.\0A\00", align 1
@.str.28 = private unnamed_addr constant [28 x i8] c"loop_ws_config_strides_dc.\0A\00", align 1
@.str.29 = private unnamed_addr constant [10 x i8] c"loop_ws.\0A\00", align 1

; Function Attrs: noinline nounwind optnone
define dso_local void @full_printMatrix(ptr noundef %m) #0 {
entry:
  %m.addr = alloca ptr, align 8
  %i = alloca i64, align 8
  %j = alloca i64, align 8
  store ptr %m, ptr %m.addr, align 8
  store i64 0, ptr %i, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.inc6, %entry
  %0 = load i64, ptr %i, align 8
  %cmp = icmp ult i64 %0, 64
  br i1 %cmp, label %for.body, label %for.end8

for.body:                                         ; preds = %for.cond
  store i64 0, ptr %j, align 8
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %1 = load i64, ptr %j, align 8
  %cmp2 = icmp ult i64 %1, 64
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %2 = load ptr, ptr %m.addr, align 8
  %3 = load i64, ptr %i, align 8
  %arrayidx = getelementptr inbounds [64 x i8], ptr %2, i64 %3
  %4 = load i64, ptr %j, align 8
  %arrayidx4 = getelementptr inbounds [64 x i8], ptr %arrayidx, i64 0, i64 %4
  %5 = load i8, ptr %arrayidx4, align 1
  %conv = sext i8 %5 to i32
  %call = call signext i32 (ptr, ...) @printf(ptr noundef @.str, i32 noundef signext %conv)
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %6 = load i64, ptr %j, align 8
  %inc = add i64 %6, 1
  store i64 %inc, ptr %j, align 8
  br label %for.cond1, !llvm.loop !5

for.end:                                          ; preds = %for.cond1
  %call5 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.1)
  br label %for.inc6

for.inc6:                                         ; preds = %for.end
  %7 = load i64, ptr %i, align 8
  %inc7 = add i64 %7, 1
  store i64 %inc7, ptr %i, align 8
  br label %for.cond, !llvm.loop !7

for.end8:                                         ; preds = %for.cond
  ret void
}

declare dso_local signext i32 @printf(ptr noundef, ...) #1

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @full_is_equal(ptr noundef %x, ptr noundef %y) #0 {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca ptr, align 8
  %y.addr = alloca ptr, align 8
  %i = alloca i64, align 8
  %j = alloca i64, align 8
  store ptr %x, ptr %x.addr, align 8
  store ptr %y, ptr %y.addr, align 8
  store i64 0, ptr %i, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.inc10, %entry
  %0 = load i64, ptr %i, align 8
  %cmp = icmp ult i64 %0, 64
  br i1 %cmp, label %for.body, label %for.end12

for.body:                                         ; preds = %for.cond
  store i64 0, ptr %j, align 8
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %1 = load i64, ptr %j, align 8
  %cmp2 = icmp ult i64 %1, 64
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %2 = load ptr, ptr %x.addr, align 8
  %3 = load i64, ptr %i, align 8
  %arrayidx = getelementptr inbounds [64 x i8], ptr %2, i64 %3
  %4 = load i64, ptr %j, align 8
  %arrayidx4 = getelementptr inbounds [64 x i8], ptr %arrayidx, i64 0, i64 %4
  %5 = load i8, ptr %arrayidx4, align 1
  %conv = sext i8 %5 to i32
  %6 = load ptr, ptr %y.addr, align 8
  %7 = load i64, ptr %i, align 8
  %arrayidx5 = getelementptr inbounds [64 x i8], ptr %6, i64 %7
  %8 = load i64, ptr %j, align 8
  %arrayidx6 = getelementptr inbounds [64 x i8], ptr %arrayidx5, i64 0, i64 %8
  %9 = load i8, ptr %arrayidx6, align 1
  %conv7 = sext i8 %9 to i32
  %cmp8 = icmp ne i32 %conv, %conv7
  br i1 %cmp8, label %if.then, label %if.end

if.then:                                          ; preds = %for.body3
  store i32 0, ptr %retval, align 4
  br label %return

if.end:                                           ; preds = %for.body3
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %10 = load i64, ptr %j, align 8
  %inc = add i64 %10, 1
  store i64 %inc, ptr %j, align 8
  br label %for.cond1, !llvm.loop !8

for.end:                                          ; preds = %for.cond1
  br label %for.inc10

for.inc10:                                        ; preds = %for.end
  %11 = load i64, ptr %i, align 8
  %inc11 = add i64 %11, 1
  store i64 %inc11, ptr %i, align 8
  br label %for.cond, !llvm.loop !9

for.end12:                                        ; preds = %for.cond
  store i32 1, ptr %retval, align 4
  br label %return

return:                                           ; preds = %for.end12, %if.then
  %12 = load i32, ptr %retval, align 4
  ret i32 %12
}

; Function Attrs: noinline nounwind optnone
define dso_local void @full_matmul(ptr noundef %A, ptr noundef %B, ptr noundef %D, ptr noundef %C_full) #0 {
entry:
  %A.addr = alloca ptr, align 8
  %B.addr = alloca ptr, align 8
  %D.addr = alloca ptr, align 8
  %C_full.addr = alloca ptr, align 8
  %r = alloca i64, align 8
  %c = alloca i64, align 8
  %k = alloca i64, align 8
  store ptr %A, ptr %A.addr, align 8
  store ptr %B, ptr %B.addr, align 8
  store ptr %D, ptr %D.addr, align 8
  store ptr %C_full, ptr %C_full.addr, align 8
  store i64 0, ptr %r, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.inc23, %entry
  %0 = load i64, ptr %r, align 8
  %cmp = icmp ult i64 %0, 64
  br i1 %cmp, label %for.body, label %for.end25

for.body:                                         ; preds = %for.cond
  store i64 0, ptr %c, align 8
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc20, %for.body
  %1 = load i64, ptr %c, align 8
  %cmp2 = icmp ult i64 %1, 64
  br i1 %cmp2, label %for.body3, label %for.end22

for.body3:                                        ; preds = %for.cond1
  %2 = load ptr, ptr %D.addr, align 8
  %3 = load i64, ptr %r, align 8
  %arrayidx = getelementptr inbounds [64 x i32], ptr %2, i64 %3
  %4 = load i64, ptr %c, align 8
  %arrayidx4 = getelementptr inbounds [64 x i32], ptr %arrayidx, i64 0, i64 %4
  %5 = load i32, ptr %arrayidx4, align 4
  %conv = sext i32 %5 to i64
  %6 = load ptr, ptr %C_full.addr, align 8
  %7 = load i64, ptr %r, align 8
  %arrayidx5 = getelementptr inbounds [64 x i64], ptr %6, i64 %7
  %8 = load i64, ptr %c, align 8
  %arrayidx6 = getelementptr inbounds [64 x i64], ptr %arrayidx5, i64 0, i64 %8
  store i64 %conv, ptr %arrayidx6, align 8
  store i64 0, ptr %k, align 8
  br label %for.cond7

for.cond7:                                        ; preds = %for.inc, %for.body3
  %9 = load i64, ptr %k, align 8
  %cmp8 = icmp ult i64 %9, 64
  br i1 %cmp8, label %for.body10, label %for.end

for.body10:                                       ; preds = %for.cond7
  %10 = load ptr, ptr %A.addr, align 8
  %11 = load i64, ptr %r, align 8
  %arrayidx11 = getelementptr inbounds [64 x i8], ptr %10, i64 %11
  %12 = load i64, ptr %k, align 8
  %arrayidx12 = getelementptr inbounds [64 x i8], ptr %arrayidx11, i64 0, i64 %12
  %13 = load i8, ptr %arrayidx12, align 1
  %conv13 = sext i8 %13 to i32
  %14 = load ptr, ptr %B.addr, align 8
  %15 = load i64, ptr %k, align 8
  %arrayidx14 = getelementptr inbounds [64 x i8], ptr %14, i64 %15
  %16 = load i64, ptr %c, align 8
  %arrayidx15 = getelementptr inbounds [64 x i8], ptr %arrayidx14, i64 0, i64 %16
  %17 = load i8, ptr %arrayidx15, align 1
  %conv16 = sext i8 %17 to i32
  %mul = mul nsw i32 %conv13, %conv16
  %conv17 = sext i32 %mul to i64
  %18 = load ptr, ptr %C_full.addr, align 8
  %19 = load i64, ptr %r, align 8
  %arrayidx18 = getelementptr inbounds [64 x i64], ptr %18, i64 %19
  %20 = load i64, ptr %c, align 8
  %arrayidx19 = getelementptr inbounds [64 x i64], ptr %arrayidx18, i64 0, i64 %20
  %21 = load i64, ptr %arrayidx19, align 8
  %add = add nsw i64 %21, %conv17
  store i64 %add, ptr %arrayidx19, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body10
  %22 = load i64, ptr %k, align 8
  %inc = add i64 %22, 1
  store i64 %inc, ptr %k, align 8
  br label %for.cond7, !llvm.loop !10

for.end:                                          ; preds = %for.cond7
  br label %for.inc20

for.inc20:                                        ; preds = %for.end
  %23 = load i64, ptr %c, align 8
  %inc21 = add i64 %23, 1
  store i64 %inc21, ptr %c, align 8
  br label %for.cond1, !llvm.loop !11

for.end22:                                        ; preds = %for.cond1
  br label %for.inc23

for.inc23:                                        ; preds = %for.end22
  %24 = load i64, ptr %r, align 8
  %inc24 = add i64 %24, 1
  store i64 %inc24, ptr %r, align 8
  br label %for.cond, !llvm.loop !12

for.end25:                                        ; preds = %for.cond
  ret void
}

; Function Attrs: noinline nounwind optnone
define dso_local void @full_matscale(ptr noundef %full, ptr noundef %out, float noundef %scale) #0 {
entry:
  %full.addr = alloca ptr, align 8
  %out.addr = alloca ptr, align 8
  %scale.addr = alloca float, align 4
  %r = alloca i64, align 8
  %c = alloca i64, align 8
  %scaled = alloca i64, align 8
  %y = alloca float, align 4
  %x_ = alloca float, align 4
  %i = alloca i64, align 8
  %next = alloca i64, align 8
  %rem = alloca float, align 4
  %result = alloca float, align 4
  %tmp = alloca float, align 4
  %tmp39 = alloca i32, align 4
  %elem = alloca i64, align 8
  store ptr %full, ptr %full.addr, align 8
  store ptr %out, ptr %out.addr, align 8
  store float %scale, ptr %scale.addr, align 4
  store i64 0, ptr %r, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.inc69, %entry
  %0 = load i64, ptr %r, align 8
  %cmp = icmp ult i64 %0, 64
  br i1 %cmp, label %for.body, label %for.end71

for.body:                                         ; preds = %for.cond
  store i64 0, ptr %c, align 8
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %1 = load i64, ptr %c, align 8
  %cmp2 = icmp ult i64 %1, 64
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %2 = load ptr, ptr %full.addr, align 8
  %3 = load i64, ptr %r, align 8
  %arrayidx = getelementptr inbounds [64 x i64], ptr %2, i64 %3
  %4 = load i64, ptr %c, align 8
  %arrayidx4 = getelementptr inbounds [64 x i64], ptr %arrayidx, i64 0, i64 %4
  %5 = load i64, ptr %arrayidx4, align 8
  %conv = sitofp i64 %5 to float
  %6 = load float, ptr %scale.addr, align 4
  %mul = fmul float %conv, %6
  store float %mul, ptr %x_, align 4
  %7 = load float, ptr %x_, align 4
  %conv5 = fptosi float %7 to i64
  store i64 %conv5, ptr %i, align 8
  %8 = load float, ptr %x_, align 4
  %cmp6 = fcmp olt float %8, 0.000000e+00
  br i1 %cmp6, label %cond.true, label %cond.false

cond.true:                                        ; preds = %for.body3
  %9 = load float, ptr %x_, align 4
  %sub = fsub float %9, 1.000000e+00
  br label %cond.end

cond.false:                                       ; preds = %for.body3
  %10 = load float, ptr %x_, align 4
  %add = fadd float %10, 1.000000e+00
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi float [ %sub, %cond.true ], [ %add, %cond.false ]
  %conv8 = fptosi float %cond to i64
  store i64 %conv8, ptr %next, align 8
  %11 = load float, ptr %x_, align 4
  %12 = load i64, ptr %i, align 8
  %conv9 = sitofp i64 %12 to float
  %sub10 = fsub float %11, %conv9
  store float %sub10, ptr %rem, align 4
  %13 = load float, ptr %rem, align 4
  %cmp11 = fcmp olt float %13, 0.000000e+00
  br i1 %cmp11, label %cond.true13, label %cond.false14

cond.true13:                                      ; preds = %cond.end
  %14 = load float, ptr %rem, align 4
  %fneg = fneg float %14
  br label %cond.end15

cond.false14:                                     ; preds = %cond.end
  %15 = load float, ptr %rem, align 4
  br label %cond.end15

cond.end15:                                       ; preds = %cond.false14, %cond.true13
  %cond16 = phi float [ %fneg, %cond.true13 ], [ %15, %cond.false14 ]
  store float %cond16, ptr %rem, align 4
  %16 = load float, ptr %rem, align 4
  %conv17 = fpext float %16 to double
  %cmp18 = fcmp olt double %conv17, 5.000000e-01
  br i1 %cmp18, label %cond.true20, label %cond.false21

cond.true20:                                      ; preds = %cond.end15
  %17 = load i64, ptr %i, align 8
  br label %cond.end36

cond.false21:                                     ; preds = %cond.end15
  %18 = load float, ptr %rem, align 4
  %conv22 = fpext float %18 to double
  %cmp23 = fcmp ogt double %conv22, 5.000000e-01
  br i1 %cmp23, label %cond.true25, label %cond.false26

cond.true25:                                      ; preds = %cond.false21
  %19 = load i64, ptr %next, align 8
  br label %cond.end34

cond.false26:                                     ; preds = %cond.false21
  %20 = load i64, ptr %i, align 8
  %rem27 = srem i64 %20, 2
  %cmp28 = icmp eq i64 %rem27, 0
  br i1 %cmp28, label %cond.true30, label %cond.false31

cond.true30:                                      ; preds = %cond.false26
  %21 = load i64, ptr %i, align 8
  br label %cond.end32

cond.false31:                                     ; preds = %cond.false26
  %22 = load i64, ptr %next, align 8
  br label %cond.end32

cond.end32:                                       ; preds = %cond.false31, %cond.true30
  %cond33 = phi i64 [ %21, %cond.true30 ], [ %22, %cond.false31 ]
  br label %cond.end34

cond.end34:                                       ; preds = %cond.end32, %cond.true25
  %cond35 = phi i64 [ %19, %cond.true25 ], [ %cond33, %cond.end32 ]
  br label %cond.end36

cond.end36:                                       ; preds = %cond.end34, %cond.true20
  %cond37 = phi i64 [ %17, %cond.true20 ], [ %cond35, %cond.end34 ]
  %conv38 = sitofp i64 %cond37 to float
  store float %conv38, ptr %result, align 4
  %23 = load float, ptr %result, align 4
  store float %23, ptr %tmp, align 4
  %24 = load float, ptr %tmp, align 4
  store float %24, ptr %y, align 4
  %25 = load float, ptr %y, align 4
  %cmp40 = fcmp ogt float %25, 1.270000e+02
  br i1 %cmp40, label %cond.true42, label %cond.false43

cond.true42:                                      ; preds = %cond.end36
  br label %cond.end51

cond.false43:                                     ; preds = %cond.end36
  %26 = load float, ptr %y, align 4
  %cmp44 = fcmp olt float %26, -1.280000e+02
  br i1 %cmp44, label %cond.true46, label %cond.false47

cond.true46:                                      ; preds = %cond.false43
  br label %cond.end49

cond.false47:                                     ; preds = %cond.false43
  %27 = load float, ptr %y, align 4
  %conv48 = fptosi float %27 to i32
  br label %cond.end49

cond.end49:                                       ; preds = %cond.false47, %cond.true46
  %cond50 = phi i32 [ -128, %cond.true46 ], [ %conv48, %cond.false47 ]
  br label %cond.end51

cond.end51:                                       ; preds = %cond.end49, %cond.true42
  %cond52 = phi i32 [ 127, %cond.true42 ], [ %cond50, %cond.end49 ]
  store i32 %cond52, ptr %tmp39, align 4
  %28 = load i32, ptr %tmp39, align 4
  %conv53 = sext i32 %28 to i64
  store i64 %conv53, ptr %scaled, align 8
  %29 = load i64, ptr %scaled, align 8
  %cmp54 = icmp sgt i64 %29, 127
  br i1 %cmp54, label %cond.true56, label %cond.false57

cond.true56:                                      ; preds = %cond.end51
  br label %cond.end64

cond.false57:                                     ; preds = %cond.end51
  %30 = load i64, ptr %scaled, align 8
  %cmp58 = icmp slt i64 %30, -128
  br i1 %cmp58, label %cond.true60, label %cond.false61

cond.true60:                                      ; preds = %cond.false57
  br label %cond.end62

cond.false61:                                     ; preds = %cond.false57
  %31 = load i64, ptr %scaled, align 8
  br label %cond.end62

cond.end62:                                       ; preds = %cond.false61, %cond.true60
  %cond63 = phi i64 [ -128, %cond.true60 ], [ %31, %cond.false61 ]
  br label %cond.end64

cond.end64:                                       ; preds = %cond.end62, %cond.true56
  %cond65 = phi i64 [ 127, %cond.true56 ], [ %cond63, %cond.end62 ]
  store i64 %cond65, ptr %elem, align 8
  %32 = load i64, ptr %elem, align 8
  %conv66 = trunc i64 %32 to i8
  %33 = load ptr, ptr %out.addr, align 8
  %34 = load i64, ptr %r, align 8
  %arrayidx67 = getelementptr inbounds [64 x i8], ptr %33, i64 %34
  %35 = load i64, ptr %c, align 8
  %arrayidx68 = getelementptr inbounds [64 x i8], ptr %arrayidx67, i64 0, i64 %35
  store i8 %conv66, ptr %arrayidx68, align 1
  br label %for.inc

for.inc:                                          ; preds = %cond.end64
  %36 = load i64, ptr %c, align 8
  %inc = add i64 %36, 1
  store i64 %inc, ptr %c, align 8
  br label %for.cond1, !llvm.loop !13

for.end:                                          ; preds = %for.cond1
  br label %for.inc69

for.inc69:                                        ; preds = %for.end
  %37 = load i64, ptr %r, align 8
  %inc70 = add i64 %37, 1
  store i64 %inc70, ptr %r, align 8
  br label %for.cond, !llvm.loop !14

for.end71:                                        ; preds = %for.cond
  ret void
}

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %full_A = alloca [64 x [64 x i8]], align 1
  %full_B = alloca [64 x [64 x i8]], align 1
  %full_C = alloca [64 x [64 x i8]], align 1
  %full_D = alloca [64 x [64 x i32]], align 4
  %gold_full = alloca [64 x [64 x i64]], align 8
  %gold = alloca [64 x [64 x i8]], align 1
  %i = alloca i64, align 8
  %j = alloca i64, align 8
  %i8 = alloca i64, align 8
  %j13 = alloca i64, align 8
  %i29 = alloca i64, align 8
  %j34 = alloca i64, align 8
  %cpu_start = alloca i64, align 8
  %cpu_end = alloca i64, align 8
  %start = alloca i64, align 8
  %end = alloca i64, align 8
  store i32 0, ptr %retval, align 4
  store i64 0, ptr %i, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.inc5, %entry
  %0 = load i64, ptr %i, align 8
  %cmp = icmp ult i64 %0, 64
  br i1 %cmp, label %for.body, label %for.end7

for.body:                                         ; preds = %for.cond
  store i64 0, ptr %j, align 8
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %1 = load i64, ptr %j, align 8
  %cmp2 = icmp ult i64 %1, 64
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %call = call signext i32 @rand()
  %rem = srem i32 %call, 2
  %conv = trunc i32 %rem to i8
  %2 = load i64, ptr %i, align 8
  %arrayidx = getelementptr inbounds [64 x [64 x i8]], ptr %full_A, i64 0, i64 %2
  %3 = load i64, ptr %j, align 8
  %arrayidx4 = getelementptr inbounds [64 x i8], ptr %arrayidx, i64 0, i64 %3
  store i8 %conv, ptr %arrayidx4, align 1
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %4 = load i64, ptr %j, align 8
  %inc = add i64 %4, 1
  store i64 %inc, ptr %j, align 8
  br label %for.cond1, !llvm.loop !15

for.end:                                          ; preds = %for.cond1
  br label %for.inc5

for.inc5:                                         ; preds = %for.end
  %5 = load i64, ptr %i, align 8
  %inc6 = add i64 %5, 1
  store i64 %inc6, ptr %i, align 8
  br label %for.cond, !llvm.loop !16

for.end7:                                         ; preds = %for.cond
  store i64 0, ptr %i8, align 8
  br label %for.cond9

for.cond9:                                        ; preds = %for.inc26, %for.end7
  %6 = load i64, ptr %i8, align 8
  %cmp10 = icmp ult i64 %6, 64
  br i1 %cmp10, label %for.body12, label %for.end28

for.body12:                                       ; preds = %for.cond9
  store i64 0, ptr %j13, align 8
  br label %for.cond14

for.cond14:                                       ; preds = %for.inc23, %for.body12
  %7 = load i64, ptr %j13, align 8
  %cmp15 = icmp ult i64 %7, 64
  br i1 %cmp15, label %for.body17, label %for.end25

for.body17:                                       ; preds = %for.cond14
  %call18 = call signext i32 @rand()
  %rem19 = srem i32 %call18, 2
  %conv20 = trunc i32 %rem19 to i8
  %8 = load i64, ptr %i8, align 8
  %arrayidx21 = getelementptr inbounds [64 x [64 x i8]], ptr %full_B, i64 0, i64 %8
  %9 = load i64, ptr %j13, align 8
  %arrayidx22 = getelementptr inbounds [64 x i8], ptr %arrayidx21, i64 0, i64 %9
  store i8 %conv20, ptr %arrayidx22, align 1
  br label %for.inc23

for.inc23:                                        ; preds = %for.body17
  %10 = load i64, ptr %j13, align 8
  %inc24 = add i64 %10, 1
  store i64 %inc24, ptr %j13, align 8
  br label %for.cond14, !llvm.loop !17

for.end25:                                        ; preds = %for.cond14
  br label %for.inc26

for.inc26:                                        ; preds = %for.end25
  %11 = load i64, ptr %i8, align 8
  %inc27 = add i64 %11, 1
  store i64 %inc27, ptr %i8, align 8
  br label %for.cond9, !llvm.loop !18

for.end28:                                        ; preds = %for.cond9
  store i64 0, ptr %i29, align 8
  br label %for.cond30

for.cond30:                                       ; preds = %for.inc44, %for.end28
  %12 = load i64, ptr %i29, align 8
  %cmp31 = icmp ult i64 %12, 64
  br i1 %cmp31, label %for.body33, label %for.end46

for.body33:                                       ; preds = %for.cond30
  store i64 0, ptr %j34, align 8
  br label %for.cond35

for.cond35:                                       ; preds = %for.inc41, %for.body33
  %13 = load i64, ptr %j34, align 8
  %cmp36 = icmp ult i64 %13, 64
  br i1 %cmp36, label %for.body38, label %for.end43

for.body38:                                       ; preds = %for.cond35
  %14 = load i64, ptr %i29, align 8
  %arrayidx39 = getelementptr inbounds [64 x [64 x i32]], ptr %full_D, i64 0, i64 %14
  %15 = load i64, ptr %j34, align 8
  %arrayidx40 = getelementptr inbounds [64 x i32], ptr %arrayidx39, i64 0, i64 %15
  store i32 0, ptr %arrayidx40, align 4
  br label %for.inc41

for.inc41:                                        ; preds = %for.body38
  %16 = load i64, ptr %j34, align 8
  %inc42 = add i64 %16, 1
  store i64 %inc42, ptr %j34, align 8
  br label %for.cond35, !llvm.loop !19

for.end43:                                        ; preds = %for.cond35
  br label %for.inc44

for.inc44:                                        ; preds = %for.end43
  %17 = load i64, ptr %i29, align 8
  %inc45 = add i64 %17, 1
  store i64 %inc45, ptr %i29, align 8
  br label %for.cond30, !llvm.loop !20

for.end46:                                        ; preds = %for.cond30
  %call47 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.2)
  %call48 = call i64 @read_cycles()
  store i64 %call48, ptr %cpu_start, align 8
  %arraydecay = getelementptr inbounds [64 x [64 x i8]], ptr %full_A, i64 0, i64 0
  %arraydecay49 = getelementptr inbounds [64 x [64 x i8]], ptr %full_B, i64 0, i64 0
  %arraydecay50 = getelementptr inbounds [64 x [64 x i32]], ptr %full_D, i64 0, i64 0
  %arraydecay51 = getelementptr inbounds [64 x [64 x i64]], ptr %gold_full, i64 0, i64 0
  call void @full_matmul(ptr noundef %arraydecay, ptr noundef %arraydecay49, ptr noundef %arraydecay50, ptr noundef %arraydecay51)
  %call52 = call i64 @read_cycles()
  store i64 %call52, ptr %cpu_end, align 8
  %18 = load i64, ptr %cpu_end, align 8
  %19 = load i64, ptr %cpu_start, align 8
  %sub = sub i64 %18, %19
  %call53 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.3, i64 noundef %sub)
  %arraydecay54 = getelementptr inbounds [64 x [64 x i64]], ptr %gold_full, i64 0, i64 0
  %arraydecay55 = getelementptr inbounds [64 x [64 x i8]], ptr %gold, i64 0, i64 0
  call void @full_matscale(ptr noundef %arraydecay54, ptr noundef %arraydecay55, float noundef 1.000000e+00)
  %call56 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.4)
  %call57 = call i64 @read_cycles()
  store i64 %call57, ptr %start, align 8
  %arraydecay58 = getelementptr inbounds [64 x [64 x i8]], ptr %full_A, i64 0, i64 0
  %arraydecay59 = getelementptr inbounds [64 x [64 x i8]], ptr %full_B, i64 0, i64 0
  %arraydecay60 = getelementptr inbounds [64 x [64 x i8]], ptr %full_C, i64 0, i64 0
  call void @tiled_matmul_auto(i64 noundef 64, i64 noundef 64, i64 noundef 64, ptr noundef %arraydecay58, ptr noundef %arraydecay59, ptr noundef null, ptr noundef %arraydecay60, i64 noundef 64, i64 noundef 64, i64 noundef 64, i64 noundef 64, float noundef 1.000000e+00, float noundef 1.000000e+00, i32 noundef 1, i32 noundef 0, float noundef 1.000000e+00, float noundef 0.000000e+00, i1 noundef false, i1 noundef false, i1 noundef false, i1 noundef false, i1 noundef false, i8 noundef 0, i32 noundef 1)
  %call61 = call i64 @read_cycles()
  store i64 %call61, ptr %end, align 8
  %20 = load i64, ptr %end, align 8
  %21 = load i64, ptr %start, align 8
  %sub62 = sub i64 %20, %21
  %call63 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.5, i64 noundef %sub62)
  %call64 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.6)
  %arraydecay65 = getelementptr inbounds [64 x [64 x i8]], ptr %full_C, i64 0, i64 0
  call void @full_printMatrix(ptr noundef %arraydecay65)
  %call66 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.7)
  %arraydecay67 = getelementptr inbounds [64 x [64 x i8]], ptr %gold, i64 0, i64 0
  call void @full_printMatrix(ptr noundef %arraydecay67)
  %call68 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.1)
  call void @exit(i32 noundef signext 0) #5
  unreachable
}

declare dso_local signext i32 @rand() #1

; Function Attrs: noinline nounwind optnone
define internal void @tiled_matmul_auto(i64 noundef %dim_I, i64 noundef %dim_J, i64 noundef %dim_K, ptr noundef %A, ptr noundef %B, ptr noundef %D, ptr noundef %C, i64 noundef %stride_A, i64 noundef %stride_B, i64 noundef %stride_D, i64 noundef %stride_C, float noundef %A_scale_factor, float noundef %B_scale_factor, i32 noundef %D_scale_factor, i32 noundef %act, float noundef %scale, float noundef %bert_scale, i1 noundef %repeating_bias, i1 noundef %transpose_A, i1 noundef %transpose_B, i1 noundef %full_C, i1 noundef %low_D, i8 noundef %weightA, i32 noundef %tiled_matmul_type) #0 {
entry:
  %dim_I.addr = alloca i64, align 8
  %dim_J.addr = alloca i64, align 8
  %dim_K.addr = alloca i64, align 8
  %A.addr = alloca ptr, align 8
  %B.addr = alloca ptr, align 8
  %D.addr = alloca ptr, align 8
  %C.addr = alloca ptr, align 8
  %stride_A.addr = alloca i64, align 8
  %stride_B.addr = alloca i64, align 8
  %stride_D.addr = alloca i64, align 8
  %stride_C.addr = alloca i64, align 8
  %A_scale_factor.addr = alloca float, align 4
  %B_scale_factor.addr = alloca float, align 4
  %D_scale_factor.addr = alloca i32, align 4
  %act.addr = alloca i32, align 4
  %scale.addr = alloca float, align 4
  %bert_scale.addr = alloca float, align 4
  %repeating_bias.addr = alloca i8, align 1
  %transpose_A.addr = alloca i8, align 1
  %transpose_B.addr = alloca i8, align 1
  %full_C.addr = alloca i8, align 1
  %low_D.addr = alloca i8, align 1
  %weightA.addr = alloca i8, align 1
  %tiled_matmul_type.addr = alloca i32, align 4
  %dim_I_padded = alloca i64, align 8
  %dim_J_padded = alloca i64, align 8
  %dim_K_padded = alloca i64, align 8
  %double_buffered = alloca i8, align 1
  %max_spad_rows = alloca i64, align 8
  %max_acc_rows = alloca i64, align 8
  %tile_I = alloca i64, align 8
  %tile_J = alloca i64, align 8
  %tile_K = alloca i64, align 8
  %increased = alloca i8, align 1
  store i64 %dim_I, ptr %dim_I.addr, align 8
  store i64 %dim_J, ptr %dim_J.addr, align 8
  store i64 %dim_K, ptr %dim_K.addr, align 8
  store ptr %A, ptr %A.addr, align 8
  store ptr %B, ptr %B.addr, align 8
  store ptr %D, ptr %D.addr, align 8
  store ptr %C, ptr %C.addr, align 8
  store i64 %stride_A, ptr %stride_A.addr, align 8
  store i64 %stride_B, ptr %stride_B.addr, align 8
  store i64 %stride_D, ptr %stride_D.addr, align 8
  store i64 %stride_C, ptr %stride_C.addr, align 8
  store float %A_scale_factor, ptr %A_scale_factor.addr, align 4
  store float %B_scale_factor, ptr %B_scale_factor.addr, align 4
  store i32 %D_scale_factor, ptr %D_scale_factor.addr, align 4
  store i32 %act, ptr %act.addr, align 4
  store float %scale, ptr %scale.addr, align 4
  store float %bert_scale, ptr %bert_scale.addr, align 4
  %frombool = zext i1 %repeating_bias to i8
  store i8 %frombool, ptr %repeating_bias.addr, align 1
  %frombool1 = zext i1 %transpose_A to i8
  store i8 %frombool1, ptr %transpose_A.addr, align 1
  %frombool2 = zext i1 %transpose_B to i8
  store i8 %frombool2, ptr %transpose_B.addr, align 1
  %frombool3 = zext i1 %full_C to i8
  store i8 %frombool3, ptr %full_C.addr, align 1
  %frombool4 = zext i1 %low_D to i8
  store i8 %frombool4, ptr %low_D.addr, align 1
  store i8 %weightA, ptr %weightA.addr, align 1
  store i32 %tiled_matmul_type, ptr %tiled_matmul_type.addr, align 4
  %0 = load i64, ptr %dim_I.addr, align 8
  %div = udiv i64 %0, 16
  %1 = load i64, ptr %dim_I.addr, align 8
  %rem = urem i64 %1, 16
  %cmp = icmp ne i64 %rem, 0
  %conv = zext i1 %cmp to i32
  %conv5 = sext i32 %conv to i64
  %add = add i64 %div, %conv5
  %mul = mul i64 %add, 16
  store i64 %mul, ptr %dim_I_padded, align 8
  %2 = load i64, ptr %dim_J.addr, align 8
  %div6 = udiv i64 %2, 16
  %3 = load i64, ptr %dim_J.addr, align 8
  %rem7 = urem i64 %3, 16
  %cmp8 = icmp ne i64 %rem7, 0
  %conv9 = zext i1 %cmp8 to i32
  %conv10 = sext i32 %conv9 to i64
  %add11 = add i64 %div6, %conv10
  %mul12 = mul i64 %add11, 16
  store i64 %mul12, ptr %dim_J_padded, align 8
  %4 = load i64, ptr %dim_K.addr, align 8
  %div13 = udiv i64 %4, 16
  %5 = load i64, ptr %dim_K.addr, align 8
  %rem14 = urem i64 %5, 16
  %cmp15 = icmp ne i64 %rem14, 0
  %conv16 = zext i1 %cmp15 to i32
  %conv17 = sext i32 %conv16 to i64
  %add18 = add i64 %div13, %conv17
  %mul19 = mul i64 %add18, 16
  store i64 %mul19, ptr %dim_K_padded, align 8
  %6 = load i32, ptr %tiled_matmul_type.addr, align 4
  %cmp20 = icmp eq i32 %6, 1
  %frombool22 = zext i1 %cmp20 to i8
  store i8 %frombool22, ptr %double_buffered, align 1
  %7 = load i8, ptr %double_buffered, align 1
  %tobool = trunc i8 %7 to i1
  %8 = zext i1 %tobool to i64
  %cond = select i1 %tobool, i32 8192, i32 16384
  %conv24 = sext i32 %cond to i64
  store i64 %conv24, ptr %max_spad_rows, align 8
  %9 = load i8, ptr %double_buffered, align 1
  %tobool25 = trunc i8 %9 to i1
  %10 = zext i1 %tobool25 to i64
  %cond27 = select i1 %tobool25, i32 512, i32 1024
  %conv28 = sext i32 %cond27 to i64
  store i64 %conv28, ptr %max_acc_rows, align 8
  %11 = load i32, ptr %act.addr, align 4
  %cmp29 = icmp eq i32 %11, 2
  br i1 %cmp29, label %if.then, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %entry
  %12 = load i32, ptr %act.addr, align 4
  %cmp31 = icmp eq i32 %12, 4
  br i1 %cmp31, label %if.then, label %if.else

if.then:                                          ; preds = %lor.lhs.false, %entry
  store i64 1, ptr %tile_I, align 8
  %13 = load i64, ptr %dim_J_padded, align 8
  %div33 = udiv i64 %13, 16
  store i64 %div33, ptr %tile_J, align 8
  store i64 1, ptr %tile_K, align 8
  br label %if.end109

if.else:                                          ; preds = %lor.lhs.false
  %14 = load i8, ptr %double_buffered, align 1
  %tobool34 = trunc i8 %14 to i1
  br i1 %tobool34, label %if.then35, label %if.else70

if.then35:                                        ; preds = %if.else
  %15 = load i64, ptr %dim_I_padded, align 8
  %div36 = udiv i64 %15, 16
  %call = call double @sqrt(double noundef 3.200000e+01) #6
  %conv37 = fptoui double %call to i64
  %cmp38 = icmp ult i64 %div36, %conv37
  br i1 %cmp38, label %cond.true, label %cond.false

cond.true:                                        ; preds = %if.then35
  %16 = load i64, ptr %dim_I_padded, align 8
  %div40 = udiv i64 %16, 16
  br label %cond.end

cond.false:                                       ; preds = %if.then35
  %call41 = call double @sqrt(double noundef 3.200000e+01) #6
  %conv42 = fptoui double %call41 to i64
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond43 = phi i64 [ %div40, %cond.true ], [ %conv42, %cond.false ]
  store i64 %cond43, ptr %tile_I, align 8
  %17 = load i64, ptr %dim_J_padded, align 8
  %div44 = udiv i64 %17, 16
  %call45 = call double @sqrt(double noundef 3.200000e+01) #6
  %conv46 = fptoui double %call45 to i64
  %cmp47 = icmp ult i64 %div44, %conv46
  br i1 %cmp47, label %cond.true49, label %cond.false51

cond.true49:                                      ; preds = %cond.end
  %18 = load i64, ptr %dim_J_padded, align 8
  %div50 = udiv i64 %18, 16
  br label %cond.end54

cond.false51:                                     ; preds = %cond.end
  %call52 = call double @sqrt(double noundef 3.200000e+01) #6
  %conv53 = fptoui double %call52 to i64
  br label %cond.end54

cond.end54:                                       ; preds = %cond.false51, %cond.true49
  %cond55 = phi i64 [ %div50, %cond.true49 ], [ %conv53, %cond.false51 ]
  store i64 %cond55, ptr %tile_J, align 8
  %19 = load i64, ptr %dim_K_padded, align 8
  %div56 = udiv i64 %19, 16
  %call57 = call double @sqrt(double noundef 3.200000e+01) #6
  %conv58 = fptoui double %call57 to i64
  %div59 = udiv i64 256, %conv58
  %cmp60 = icmp ult i64 %div56, %div59
  br i1 %cmp60, label %cond.true62, label %cond.false64

cond.true62:                                      ; preds = %cond.end54
  %20 = load i64, ptr %dim_K_padded, align 8
  %div63 = udiv i64 %20, 16
  br label %cond.end68

cond.false64:                                     ; preds = %cond.end54
  %call65 = call double @sqrt(double noundef 3.200000e+01) #6
  %conv66 = fptoui double %call65 to i64
  %div67 = udiv i64 256, %conv66
  br label %cond.end68

cond.end68:                                       ; preds = %cond.false64, %cond.true62
  %cond69 = phi i64 [ %div63, %cond.true62 ], [ %div67, %cond.false64 ]
  store i64 %cond69, ptr %tile_K, align 8
  br label %if.end

if.else70:                                        ; preds = %if.else
  %21 = load i64, ptr %dim_I_padded, align 8
  %div71 = udiv i64 %21, 16
  %call72 = call double @sqrt(double noundef 6.400000e+01) #6
  %conv73 = fptoui double %call72 to i64
  %cmp74 = icmp ult i64 %div71, %conv73
  br i1 %cmp74, label %cond.true76, label %cond.false78

cond.true76:                                      ; preds = %if.else70
  %22 = load i64, ptr %dim_I_padded, align 8
  %div77 = udiv i64 %22, 16
  br label %cond.end81

cond.false78:                                     ; preds = %if.else70
  %call79 = call double @sqrt(double noundef 6.400000e+01) #6
  %conv80 = fptoui double %call79 to i64
  br label %cond.end81

cond.end81:                                       ; preds = %cond.false78, %cond.true76
  %cond82 = phi i64 [ %div77, %cond.true76 ], [ %conv80, %cond.false78 ]
  store i64 %cond82, ptr %tile_I, align 8
  %23 = load i64, ptr %dim_J_padded, align 8
  %div83 = udiv i64 %23, 16
  %call84 = call double @sqrt(double noundef 6.400000e+01) #6
  %conv85 = fptoui double %call84 to i64
  %cmp86 = icmp ult i64 %div83, %conv85
  br i1 %cmp86, label %cond.true88, label %cond.false90

cond.true88:                                      ; preds = %cond.end81
  %24 = load i64, ptr %dim_J_padded, align 8
  %div89 = udiv i64 %24, 16
  br label %cond.end93

cond.false90:                                     ; preds = %cond.end81
  %call91 = call double @sqrt(double noundef 6.400000e+01) #6
  %conv92 = fptoui double %call91 to i64
  br label %cond.end93

cond.end93:                                       ; preds = %cond.false90, %cond.true88
  %cond94 = phi i64 [ %div89, %cond.true88 ], [ %conv92, %cond.false90 ]
  store i64 %cond94, ptr %tile_J, align 8
  %25 = load i64, ptr %dim_K_padded, align 8
  %div95 = udiv i64 %25, 16
  %call96 = call double @sqrt(double noundef 6.400000e+01) #6
  %conv97 = fptoui double %call96 to i64
  %div98 = udiv i64 512, %conv97
  %cmp99 = icmp ult i64 %div95, %div98
  br i1 %cmp99, label %cond.true101, label %cond.false103

cond.true101:                                     ; preds = %cond.end93
  %26 = load i64, ptr %dim_K_padded, align 8
  %div102 = udiv i64 %26, 16
  br label %cond.end107

cond.false103:                                    ; preds = %cond.end93
  %call104 = call double @sqrt(double noundef 6.400000e+01) #6
  %conv105 = fptoui double %call104 to i64
  %div106 = udiv i64 512, %conv105
  br label %cond.end107

cond.end107:                                      ; preds = %cond.false103, %cond.true101
  %cond108 = phi i64 [ %div102, %cond.true101 ], [ %div106, %cond.false103 ]
  store i64 %cond108, ptr %tile_K, align 8
  br label %if.end

if.end:                                           ; preds = %cond.end107, %cond.end68
  br label %if.end109

if.end109:                                        ; preds = %if.end, %if.then
  br label %while.body

while.body:                                       ; preds = %if.end109, %if.end156
  store i8 0, ptr %increased, align 1
  %27 = load i64, ptr %tile_I, align 8
  %28 = load i64, ptr %tile_J, align 8
  %add110 = add i64 %28, 1
  %29 = load i64, ptr %tile_K, align 8
  %call111 = call i64 @tiled_matmul_total_spad_rows(i64 noundef %27, i64 noundef %add110, i64 noundef %29)
  %30 = load i64, ptr %max_spad_rows, align 8
  %cmp112 = icmp ule i64 %call111, %30
  br i1 %cmp112, label %land.lhs.true, label %if.end124

land.lhs.true:                                    ; preds = %while.body
  %31 = load i64, ptr %tile_I, align 8
  %32 = load i64, ptr %tile_J, align 8
  %add114 = add i64 %32, 1
  %call115 = call i64 @tiled_matmul_total_acc_rows(i64 noundef %31, i64 noundef %add114)
  %33 = load i64, ptr %max_acc_rows, align 8
  %cmp116 = icmp ule i64 %call115, %33
  br i1 %cmp116, label %land.lhs.true118, label %if.end124

land.lhs.true118:                                 ; preds = %land.lhs.true
  %34 = load i64, ptr %tile_J, align 8
  %add119 = add i64 %34, 1
  %mul120 = mul i64 %add119, 16
  %35 = load i64, ptr %dim_J_padded, align 8
  %cmp121 = icmp ule i64 %mul120, %35
  br i1 %cmp121, label %if.then123, label %if.end124

if.then123:                                       ; preds = %land.lhs.true118
  %36 = load i64, ptr %tile_J, align 8
  %inc = add i64 %36, 1
  store i64 %inc, ptr %tile_J, align 8
  store i8 1, ptr %increased, align 1
  br label %if.end124

if.end124:                                        ; preds = %if.then123, %land.lhs.true118, %land.lhs.true, %while.body
  %37 = load i64, ptr %tile_I, align 8
  %add125 = add i64 %37, 1
  %38 = load i64, ptr %tile_J, align 8
  %39 = load i64, ptr %tile_K, align 8
  %call126 = call i64 @tiled_matmul_total_spad_rows(i64 noundef %add125, i64 noundef %38, i64 noundef %39)
  %40 = load i64, ptr %max_spad_rows, align 8
  %cmp127 = icmp ule i64 %call126, %40
  br i1 %cmp127, label %land.lhs.true129, label %if.end141

land.lhs.true129:                                 ; preds = %if.end124
  %41 = load i64, ptr %tile_I, align 8
  %add130 = add i64 %41, 1
  %42 = load i64, ptr %tile_J, align 8
  %call131 = call i64 @tiled_matmul_total_acc_rows(i64 noundef %add130, i64 noundef %42)
  %43 = load i64, ptr %max_acc_rows, align 8
  %cmp132 = icmp ule i64 %call131, %43
  br i1 %cmp132, label %land.lhs.true134, label %if.end141

land.lhs.true134:                                 ; preds = %land.lhs.true129
  %44 = load i64, ptr %tile_I, align 8
  %add135 = add i64 %44, 1
  %mul136 = mul i64 %add135, 16
  %45 = load i64, ptr %dim_I_padded, align 8
  %cmp137 = icmp ule i64 %mul136, %45
  br i1 %cmp137, label %if.then139, label %if.end141

if.then139:                                       ; preds = %land.lhs.true134
  %46 = load i64, ptr %tile_I, align 8
  %inc140 = add i64 %46, 1
  store i64 %inc140, ptr %tile_I, align 8
  store i8 1, ptr %increased, align 1
  br label %if.end141

if.end141:                                        ; preds = %if.then139, %land.lhs.true134, %land.lhs.true129, %if.end124
  %47 = load i64, ptr %tile_I, align 8
  %48 = load i64, ptr %tile_J, align 8
  %49 = load i64, ptr %tile_K, align 8
  %add142 = add i64 %49, 1
  %call143 = call i64 @tiled_matmul_total_spad_rows(i64 noundef %47, i64 noundef %48, i64 noundef %add142)
  %50 = load i64, ptr %max_spad_rows, align 8
  %cmp144 = icmp ule i64 %call143, %50
  br i1 %cmp144, label %land.lhs.true146, label %if.end153

land.lhs.true146:                                 ; preds = %if.end141
  %51 = load i64, ptr %tile_K, align 8
  %add147 = add i64 %51, 1
  %mul148 = mul i64 %add147, 16
  %52 = load i64, ptr %dim_K_padded, align 8
  %cmp149 = icmp ule i64 %mul148, %52
  br i1 %cmp149, label %if.then151, label %if.end153

if.then151:                                       ; preds = %land.lhs.true146
  %53 = load i64, ptr %tile_K, align 8
  %inc152 = add i64 %53, 1
  store i64 %inc152, ptr %tile_K, align 8
  store i8 1, ptr %increased, align 1
  br label %if.end153

if.end153:                                        ; preds = %if.then151, %land.lhs.true146, %if.end141
  %54 = load i8, ptr %increased, align 1
  %tobool154 = trunc i8 %54 to i1
  br i1 %tobool154, label %if.end156, label %if.then155

if.then155:                                       ; preds = %if.end153
  br label %while.end

if.end156:                                        ; preds = %if.end153
  br label %while.body

while.end:                                        ; preds = %if.then155
  %55 = load i64, ptr %dim_I.addr, align 8
  %56 = load i64, ptr %dim_J.addr, align 8
  %57 = load i64, ptr %dim_K.addr, align 8
  %58 = load ptr, ptr %A.addr, align 8
  %59 = load ptr, ptr %B.addr, align 8
  %60 = load ptr, ptr %D.addr, align 8
  %61 = load ptr, ptr %C.addr, align 8
  %62 = load i64, ptr %stride_A.addr, align 8
  %63 = load i64, ptr %stride_B.addr, align 8
  %64 = load i64, ptr %stride_D.addr, align 8
  %65 = load i64, ptr %stride_C.addr, align 8
  %66 = load float, ptr %A_scale_factor.addr, align 4
  %67 = load float, ptr %B_scale_factor.addr, align 4
  %68 = load i32, ptr %D_scale_factor.addr, align 4
  %69 = load i32, ptr %act.addr, align 4
  %70 = load float, ptr %scale.addr, align 4
  %71 = load float, ptr %bert_scale.addr, align 4
  %72 = load i8, ptr %repeating_bias.addr, align 1
  %tobool157 = trunc i8 %72 to i1
  %73 = load i64, ptr %tile_I, align 8
  %74 = load i64, ptr %tile_J, align 8
  %75 = load i64, ptr %tile_K, align 8
  %76 = load i8, ptr %transpose_A.addr, align 1
  %tobool158 = trunc i8 %76 to i1
  %77 = load i8, ptr %transpose_B.addr, align 1
  %tobool159 = trunc i8 %77 to i1
  %78 = load i8, ptr %full_C.addr, align 1
  %tobool160 = trunc i8 %78 to i1
  %79 = load i8, ptr %low_D.addr, align 1
  %tobool161 = trunc i8 %79 to i1
  %80 = load i8, ptr %weightA.addr, align 1
  %81 = load i32, ptr %tiled_matmul_type.addr, align 4
  call void @tiled_matmul(i64 noundef %55, i64 noundef %56, i64 noundef %57, ptr noundef %58, ptr noundef %59, ptr noundef %60, ptr noundef %61, i64 noundef %62, i64 noundef %63, i64 noundef %64, i64 noundef %65, float noundef %66, float noundef %67, i32 noundef %68, i32 noundef %69, float noundef %70, float noundef %71, i1 noundef %tobool157, i64 noundef %73, i64 noundef %74, i64 noundef %75, i1 noundef %tobool158, i1 noundef %tobool159, i1 noundef %tobool160, i1 noundef %tobool161, i8 noundef %80, i32 noundef %81)
  ret void
}

; Function Attrs: noreturn
declare dso_local void @exit(i32 noundef signext) #2

; Function Attrs: noinline nounwind optnone
define internal i64 @read_cycles() #0 {
entry:
  %cycles = alloca i64, align 8
  %0 = call i64 asm sideeffect "rdcycle $0", "=r"() #6, !srcloc !21
  store i64 %0, ptr %cycles, align 8
  %1 = load i64, ptr %cycles, align 8
  ret i64 %1
}

; Function Attrs: nounwind
declare dso_local double @sqrt(double noundef) #3

; Function Attrs: noinline nounwind optnone
define internal i64 @tiled_matmul_total_spad_rows(i64 noundef %I, i64 noundef %J, i64 noundef %K) #0 {
entry:
  %I.addr = alloca i64, align 8
  %J.addr = alloca i64, align 8
  %K.addr = alloca i64, align 8
  store i64 %I, ptr %I.addr, align 8
  store i64 %J, ptr %J.addr, align 8
  store i64 %K, ptr %K.addr, align 8
  %0 = load i64, ptr %I.addr, align 8
  %1 = load i64, ptr %K.addr, align 8
  %mul = mul i64 %0, %1
  %2 = load i64, ptr %K.addr, align 8
  %3 = load i64, ptr %J.addr, align 8
  %mul1 = mul i64 %2, %3
  %add = add i64 %mul, %mul1
  %mul2 = mul i64 %add, 16
  ret i64 %mul2
}

; Function Attrs: noinline nounwind optnone
define internal i64 @tiled_matmul_total_acc_rows(i64 noundef %I, i64 noundef %J) #0 {
entry:
  %I.addr = alloca i64, align 8
  %J.addr = alloca i64, align 8
  store i64 %I, ptr %I.addr, align 8
  store i64 %J, ptr %J.addr, align 8
  %0 = load i64, ptr %I.addr, align 8
  %1 = load i64, ptr %J.addr, align 8
  %mul = mul i64 %0, %1
  %mul1 = mul i64 %mul, 16
  ret i64 %mul1
}

; Function Attrs: noinline nounwind optnone
define internal void @tiled_matmul(i64 noundef %dim_I, i64 noundef %dim_J, i64 noundef %dim_K, ptr noundef %A, ptr noundef %B, ptr noundef %D, ptr noundef %C, i64 noundef %stride_A, i64 noundef %stride_B, i64 noundef %stride_D, i64 noundef %stride_C, float noundef %A_scale_factor, float noundef %B_scale_factor, i32 noundef %D_scale_factor, i32 noundef %act, float noundef %scale, float noundef %bert_scale, i1 noundef %repeating_bias, i64 noundef %tile_I, i64 noundef %tile_J, i64 noundef %tile_K, i1 noundef %transpose_A, i1 noundef %transpose_B, i1 noundef %full_C, i1 noundef %low_D, i8 noundef %weightA, i32 noundef %tiled_matmul_type) #0 {
entry:
  %dim_I.addr = alloca i64, align 8
  %dim_J.addr = alloca i64, align 8
  %dim_K.addr = alloca i64, align 8
  %A.addr = alloca ptr, align 8
  %B.addr = alloca ptr, align 8
  %D.addr = alloca ptr, align 8
  %C.addr = alloca ptr, align 8
  %stride_A.addr = alloca i64, align 8
  %stride_B.addr = alloca i64, align 8
  %stride_D.addr = alloca i64, align 8
  %stride_C.addr = alloca i64, align 8
  %A_scale_factor.addr = alloca float, align 4
  %B_scale_factor.addr = alloca float, align 4
  %D_scale_factor.addr = alloca i32, align 4
  %act.addr = alloca i32, align 4
  %scale.addr = alloca float, align 4
  %bert_scale.addr = alloca float, align 4
  %repeating_bias.addr = alloca i8, align 1
  %tile_I.addr = alloca i64, align 8
  %tile_J.addr = alloca i64, align 8
  %tile_K.addr = alloca i64, align 8
  %transpose_A.addr = alloca i8, align 1
  %transpose_B.addr = alloca i8, align 1
  %full_C.addr = alloca i8, align 1
  %low_D.addr = alloca i8, align 1
  %weightA.addr = alloca i8, align 1
  %tiled_matmul_type.addr = alloca i32, align 4
  %dim_I_padded = alloca i64, align 8
  %dim_J_padded = alloca i64, align 8
  %dim_K_padded = alloca i64, align 8
  %double_buffered = alloca i8, align 1
  %total_spad_size = alloca i64, align 8
  %total_acc_size = alloca i64, align 8
  %total_spad_rows = alloca i64, align 8
  %total_acc_rows = alloca i64, align 8
  %matmul_type_str = alloca [3 x [4 x i8]], align 1
  store i64 %dim_I, ptr %dim_I.addr, align 8
  store i64 %dim_J, ptr %dim_J.addr, align 8
  store i64 %dim_K, ptr %dim_K.addr, align 8
  store ptr %A, ptr %A.addr, align 8
  store ptr %B, ptr %B.addr, align 8
  store ptr %D, ptr %D.addr, align 8
  store ptr %C, ptr %C.addr, align 8
  store i64 %stride_A, ptr %stride_A.addr, align 8
  store i64 %stride_B, ptr %stride_B.addr, align 8
  store i64 %stride_D, ptr %stride_D.addr, align 8
  store i64 %stride_C, ptr %stride_C.addr, align 8
  store float %A_scale_factor, ptr %A_scale_factor.addr, align 4
  store float %B_scale_factor, ptr %B_scale_factor.addr, align 4
  store i32 %D_scale_factor, ptr %D_scale_factor.addr, align 4
  store i32 %act, ptr %act.addr, align 4
  store float %scale, ptr %scale.addr, align 4
  store float %bert_scale, ptr %bert_scale.addr, align 4
  %frombool = zext i1 %repeating_bias to i8
  store i8 %frombool, ptr %repeating_bias.addr, align 1
  store i64 %tile_I, ptr %tile_I.addr, align 8
  store i64 %tile_J, ptr %tile_J.addr, align 8
  store i64 %tile_K, ptr %tile_K.addr, align 8
  %frombool1 = zext i1 %transpose_A to i8
  store i8 %frombool1, ptr %transpose_A.addr, align 1
  %frombool2 = zext i1 %transpose_B to i8
  store i8 %frombool2, ptr %transpose_B.addr, align 1
  %frombool3 = zext i1 %full_C to i8
  store i8 %frombool3, ptr %full_C.addr, align 1
  %frombool4 = zext i1 %low_D to i8
  store i8 %frombool4, ptr %low_D.addr, align 1
  store i8 %weightA, ptr %weightA.addr, align 1
  store i32 %tiled_matmul_type, ptr %tiled_matmul_type.addr, align 4
  %0 = load i64, ptr %tile_I.addr, align 8
  %cmp = icmp ule i64 %0, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %call = call signext i32 (ptr, ...) @printf(ptr noundef @.str.8)
  call void @exit(i32 noundef signext 1) #5
  unreachable

if.else:                                          ; preds = %entry
  %1 = load i64, ptr %tile_J.addr, align 8
  %cmp5 = icmp ule i64 %1, 0
  br i1 %cmp5, label %if.then6, label %if.else8

if.then6:                                         ; preds = %if.else
  %call7 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.9)
  call void @exit(i32 noundef signext 1) #5
  unreachable

if.else8:                                         ; preds = %if.else
  %2 = load i64, ptr %tile_K.addr, align 8
  %cmp9 = icmp ule i64 %2, 0
  br i1 %cmp9, label %if.then10, label %if.end

if.then10:                                        ; preds = %if.else8
  %call11 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.10)
  call void @exit(i32 noundef signext 1) #5
  unreachable

if.end:                                           ; preds = %if.else8
  br label %if.end12

if.end12:                                         ; preds = %if.end
  br label %if.end13

if.end13:                                         ; preds = %if.end12
  %3 = load i64, ptr %dim_I.addr, align 8
  %div = udiv i64 %3, 16
  %4 = load i64, ptr %dim_I.addr, align 8
  %rem = urem i64 %4, 16
  %cmp14 = icmp ne i64 %rem, 0
  %conv = zext i1 %cmp14 to i32
  %conv15 = sext i32 %conv to i64
  %add = add i64 %div, %conv15
  %mul = mul i64 %add, 16
  store i64 %mul, ptr %dim_I_padded, align 8
  %5 = load i64, ptr %dim_J.addr, align 8
  %div16 = udiv i64 %5, 16
  %6 = load i64, ptr %dim_J.addr, align 8
  %rem17 = urem i64 %6, 16
  %cmp18 = icmp ne i64 %rem17, 0
  %conv19 = zext i1 %cmp18 to i32
  %conv20 = sext i32 %conv19 to i64
  %add21 = add i64 %div16, %conv20
  %mul22 = mul i64 %add21, 16
  store i64 %mul22, ptr %dim_J_padded, align 8
  %7 = load i64, ptr %dim_K.addr, align 8
  %div23 = udiv i64 %7, 16
  %8 = load i64, ptr %dim_K.addr, align 8
  %rem24 = urem i64 %8, 16
  %cmp25 = icmp ne i64 %rem24, 0
  %conv26 = zext i1 %cmp25 to i32
  %conv27 = sext i32 %conv26 to i64
  %add28 = add i64 %div23, %conv27
  %mul29 = mul i64 %add28, 16
  store i64 %mul29, ptr %dim_K_padded, align 8
  %9 = load i64, ptr %tile_I.addr, align 8
  %mul30 = mul i64 %9, 16
  %10 = load i64, ptr %dim_I_padded, align 8
  %cmp31 = icmp ugt i64 %mul30, %10
  br i1 %cmp31, label %if.then33, label %if.else35

if.then33:                                        ; preds = %if.end13
  %call34 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.11)
  call void @exit(i32 noundef signext 1) #5
  unreachable

if.else35:                                        ; preds = %if.end13
  %11 = load i64, ptr %tile_J.addr, align 8
  %mul36 = mul i64 %11, 16
  %12 = load i64, ptr %dim_J_padded, align 8
  %cmp37 = icmp ugt i64 %mul36, %12
  br i1 %cmp37, label %if.then39, label %if.else41

if.then39:                                        ; preds = %if.else35
  %call40 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.12)
  call void @exit(i32 noundef signext 1) #5
  unreachable

if.else41:                                        ; preds = %if.else35
  %13 = load i64, ptr %tile_K.addr, align 8
  %mul42 = mul i64 %13, 16
  %14 = load i64, ptr %dim_K_padded, align 8
  %cmp43 = icmp ugt i64 %mul42, %14
  br i1 %cmp43, label %if.then45, label %if.end47

if.then45:                                        ; preds = %if.else41
  %call46 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.13)
  call void @exit(i32 noundef signext 1) #5
  unreachable

if.end47:                                         ; preds = %if.else41
  br label %if.end48

if.end48:                                         ; preds = %if.end47
  br label %if.end49

if.end49:                                         ; preds = %if.end48
  %15 = load i32, ptr %tiled_matmul_type.addr, align 4
  %cmp50 = icmp eq i32 %15, 1
  %frombool52 = zext i1 %cmp50 to i8
  store i8 %frombool52, ptr %double_buffered, align 1
  %16 = load i8, ptr %double_buffered, align 1
  %tobool = trunc i8 %16 to i1
  %17 = zext i1 %tobool to i64
  %cond = select i1 %tobool, i32 8192, i32 16384
  %conv54 = sext i32 %cond to i64
  store i64 %conv54, ptr %total_spad_size, align 8
  %18 = load i8, ptr %double_buffered, align 1
  %tobool55 = trunc i8 %18 to i1
  %19 = zext i1 %tobool55 to i64
  %cond57 = select i1 %tobool55, i32 512, i32 1024
  %conv58 = sext i32 %cond57 to i64
  store i64 %conv58, ptr %total_acc_size, align 8
  %20 = load i64, ptr %tile_I.addr, align 8
  %21 = load i64, ptr %tile_K.addr, align 8
  %mul59 = mul i64 %20, %21
  %mul60 = mul i64 %mul59, 16
  %22 = load i64, ptr %tile_K.addr, align 8
  %23 = load i64, ptr %tile_J.addr, align 8
  %mul61 = mul i64 %22, %23
  %mul62 = mul i64 %mul61, 16
  %add63 = add i64 %mul60, %mul62
  store i64 %add63, ptr %total_spad_rows, align 8
  %24 = load i64, ptr %total_spad_rows, align 8
  %25 = load i64, ptr %total_spad_size, align 8
  %cmp64 = icmp ugt i64 %24, %25
  br i1 %cmp64, label %if.then66, label %if.end68

if.then66:                                        ; preds = %if.end49
  %call67 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.14)
  call void @exit(i32 noundef signext 1) #5
  unreachable

if.end68:                                         ; preds = %if.end49
  %26 = load i64, ptr %tile_I.addr, align 8
  %27 = load i64, ptr %tile_J.addr, align 8
  %mul69 = mul i64 %26, %27
  %mul70 = mul i64 %mul69, 16
  store i64 %mul70, ptr %total_acc_rows, align 8
  %28 = load i64, ptr %total_acc_rows, align 8
  %29 = load i64, ptr %total_acc_size, align 8
  %cmp71 = icmp ugt i64 %28, %29
  br i1 %cmp71, label %if.then73, label %if.end75

if.then73:                                        ; preds = %if.end68
  %call74 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.15)
  call void @exit(i32 noundef signext 1) #5
  unreachable

if.end75:                                         ; preds = %if.end68
  %30 = load i64, ptr %tile_I.addr, align 8
  %cmp76 = icmp ugt i64 %30, 65535
  br i1 %cmp76, label %if.then83, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %if.end75
  %31 = load i64, ptr %tile_J.addr, align 8
  %cmp78 = icmp ugt i64 %31, 65535
  br i1 %cmp78, label %if.then83, label %lor.lhs.false80

lor.lhs.false80:                                  ; preds = %lor.lhs.false
  %32 = load i64, ptr %tile_K.addr, align 8
  %cmp81 = icmp ugt i64 %32, 65535
  br i1 %cmp81, label %if.then83, label %if.end85

if.then83:                                        ; preds = %lor.lhs.false80, %lor.lhs.false, %if.end75
  %call84 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.16)
  call void @exit(i32 noundef signext 1) #5
  unreachable

if.end85:                                         ; preds = %lor.lhs.false80
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %matmul_type_str, ptr align 1 @__const.tiled_matmul.matmul_type_str, i64 12, i1 false)
  %33 = load i32, ptr %tiled_matmul_type.addr, align 4
  %cmp86 = icmp eq i32 %33, 0
  br i1 %cmp86, label %land.lhs.true, label %lor.lhs.false93

land.lhs.true:                                    ; preds = %if.end85
  %34 = load i8, ptr %transpose_A.addr, align 1
  %tobool88 = trunc i8 %34 to i1
  br i1 %tobool88, label %if.then102, label %lor.lhs.false90

lor.lhs.false90:                                  ; preds = %land.lhs.true
  %35 = load i8, ptr %transpose_B.addr, align 1
  %tobool91 = trunc i8 %35 to i1
  br i1 %tobool91, label %if.then102, label %lor.lhs.false93

lor.lhs.false93:                                  ; preds = %lor.lhs.false90, %if.end85
  %36 = load i32, ptr %tiled_matmul_type.addr, align 4
  %cmp94 = icmp eq i32 %36, 1
  br i1 %cmp94, label %land.lhs.true96, label %if.end108

land.lhs.true96:                                  ; preds = %lor.lhs.false93
  %37 = load i8, ptr %transpose_A.addr, align 1
  %tobool97 = trunc i8 %37 to i1
  br i1 %tobool97, label %land.lhs.true99, label %if.end108

land.lhs.true99:                                  ; preds = %land.lhs.true96
  %38 = load i8, ptr %transpose_B.addr, align 1
  %tobool100 = trunc i8 %38 to i1
  br i1 %tobool100, label %if.then102, label %if.end108

if.then102:                                       ; preds = %land.lhs.true99, %lor.lhs.false90, %land.lhs.true
  %39 = load i32, ptr %tiled_matmul_type.addr, align 4
  %idxprom = zext i32 %39 to i64
  %arrayidx = getelementptr inbounds [3 x [4 x i8]], ptr %matmul_type_str, i64 0, i64 %idxprom
  %arraydecay = getelementptr inbounds [4 x i8], ptr %arrayidx, i64 0, i64 0
  %40 = load i8, ptr %transpose_A.addr, align 1
  %tobool103 = trunc i8 %40 to i1
  %conv104 = zext i1 %tobool103 to i32
  %41 = load i8, ptr %transpose_B.addr, align 1
  %tobool105 = trunc i8 %41 to i1
  %conv106 = zext i1 %tobool105 to i32
  %call107 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.17, ptr noundef %arraydecay, i32 noundef signext %conv104, i32 noundef signext %conv106)
  call void @exit(i32 noundef signext 1) #5
  unreachable

if.end108:                                        ; preds = %land.lhs.true99, %land.lhs.true96, %lor.lhs.false93
  %42 = load i32, ptr %tiled_matmul_type.addr, align 4
  %cmp109 = icmp eq i32 %42, 2
  br i1 %cmp109, label %land.lhs.true111, label %lor.lhs.false117

land.lhs.true111:                                 ; preds = %if.end108
  %43 = load i8, ptr %full_C.addr, align 1
  %tobool112 = trunc i8 %43 to i1
  br i1 %tobool112, label %if.then123, label %lor.lhs.false114

lor.lhs.false114:                                 ; preds = %land.lhs.true111
  %44 = load i8, ptr %low_D.addr, align 1
  %tobool115 = trunc i8 %44 to i1
  br i1 %tobool115, label %if.then123, label %lor.lhs.false117

lor.lhs.false117:                                 ; preds = %lor.lhs.false114, %if.end108
  %45 = load i32, ptr %tiled_matmul_type.addr, align 4
  %cmp118 = icmp eq i32 %45, 0
  br i1 %cmp118, label %land.lhs.true120, label %if.end132

land.lhs.true120:                                 ; preds = %lor.lhs.false117
  %46 = load i8, ptr %low_D.addr, align 1
  %tobool121 = trunc i8 %46 to i1
  br i1 %tobool121, label %if.then123, label %if.end132

if.then123:                                       ; preds = %land.lhs.true120, %lor.lhs.false114, %land.lhs.true111
  %47 = load i32, ptr %tiled_matmul_type.addr, align 4
  %idxprom124 = zext i32 %47 to i64
  %arrayidx125 = getelementptr inbounds [3 x [4 x i8]], ptr %matmul_type_str, i64 0, i64 %idxprom124
  %arraydecay126 = getelementptr inbounds [4 x i8], ptr %arrayidx125, i64 0, i64 0
  %48 = load i8, ptr %full_C.addr, align 1
  %tobool127 = trunc i8 %48 to i1
  %conv128 = zext i1 %tobool127 to i32
  %49 = load i8, ptr %low_D.addr, align 1
  %tobool129 = trunc i8 %49 to i1
  %conv130 = zext i1 %tobool129 to i32
  %call131 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.18, ptr noundef %arraydecay126, i32 noundef signext %conv128, i32 noundef signext %conv130)
  br label %if.end132

if.end132:                                        ; preds = %if.then123, %land.lhs.true120, %lor.lhs.false117
  %50 = load i32, ptr %act.addr, align 4
  %cmp133 = icmp eq i32 %50, 2
  br i1 %cmp133, label %if.then138, label %lor.lhs.false135

lor.lhs.false135:                                 ; preds = %if.end132
  %51 = load i32, ptr %act.addr, align 4
  %cmp136 = icmp eq i32 %51, 4
  br i1 %cmp136, label %if.then138, label %if.end153

if.then138:                                       ; preds = %lor.lhs.false135, %if.end132
  %52 = load i32, ptr %tiled_matmul_type.addr, align 4
  %cmp139 = icmp eq i32 %52, 0
  br i1 %cmp139, label %if.then141, label %if.end146

if.then141:                                       ; preds = %if.then138
  %53 = load i32, ptr %tiled_matmul_type.addr, align 4
  %idxprom142 = zext i32 %53 to i64
  %arrayidx143 = getelementptr inbounds [3 x [4 x i8]], ptr %matmul_type_str, i64 0, i64 %idxprom142
  %arraydecay144 = getelementptr inbounds [4 x i8], ptr %arrayidx143, i64 0, i64 0
  %54 = load i32, ptr %act.addr, align 4
  %call145 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.19, ptr noundef %arraydecay144, i32 noundef signext %54)
  br label %if.end146

if.end146:                                        ; preds = %if.then141, %if.then138
  %55 = load i64, ptr %tile_J.addr, align 8
  %mul147 = mul i64 %55, 16
  %56 = load i64, ptr %dim_J.addr, align 8
  %cmp148 = icmp ult i64 %mul147, %56
  br i1 %cmp148, label %if.then150, label %if.end152

if.then150:                                       ; preds = %if.end146
  %call151 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.20)
  br label %if.end152

if.end152:                                        ; preds = %if.then150, %if.end146
  br label %if.end153

if.end153:                                        ; preds = %if.end152, %lor.lhs.false135
  %57 = load i64, ptr %dim_I.addr, align 8
  %58 = load i64, ptr %dim_J.addr, align 8
  %59 = load i64, ptr %dim_K.addr, align 8
  %60 = load ptr, ptr %A.addr, align 8
  %61 = load ptr, ptr %B.addr, align 8
  %62 = load ptr, ptr %D.addr, align 8
  %63 = load ptr, ptr %C.addr, align 8
  %64 = load i64, ptr %stride_A.addr, align 8
  %65 = load i64, ptr %stride_B.addr, align 8
  %66 = load i64, ptr %stride_D.addr, align 8
  %67 = load i64, ptr %stride_C.addr, align 8
  %68 = load float, ptr %A_scale_factor.addr, align 4
  %69 = load float, ptr %B_scale_factor.addr, align 4
  %70 = load i32, ptr %D_scale_factor.addr, align 4
  %71 = load i64, ptr %tile_I.addr, align 8
  %72 = load i64, ptr %tile_J.addr, align 8
  %73 = load i64, ptr %tile_K.addr, align 8
  %74 = load i32, ptr %act.addr, align 4
  %75 = load float, ptr %scale.addr, align 4
  %76 = load float, ptr %bert_scale.addr, align 4
  %77 = load i8, ptr %repeating_bias.addr, align 1
  %tobool154 = trunc i8 %77 to i1
  %78 = load i8, ptr %transpose_A.addr, align 1
  %tobool155 = trunc i8 %78 to i1
  %79 = load i8, ptr %transpose_B.addr, align 1
  %tobool156 = trunc i8 %79 to i1
  %80 = load i8, ptr %full_C.addr, align 1
  %tobool157 = trunc i8 %80 to i1
  %81 = load i8, ptr %low_D.addr, align 1
  %tobool158 = trunc i8 %81 to i1
  %82 = load i8, ptr %weightA.addr, align 1
  %83 = load i32, ptr %tiled_matmul_type.addr, align 4
  call void @tiled_matmul_outer(i64 noundef %57, i64 noundef %58, i64 noundef %59, ptr noundef %60, ptr noundef %61, ptr noundef %62, ptr noundef %63, i64 noundef %64, i64 noundef %65, i64 noundef %66, i64 noundef %67, float noundef %68, float noundef %69, i32 noundef %70, i64 noundef %71, i64 noundef %72, i64 noundef %73, i32 noundef %74, float noundef %75, float noundef %76, i1 noundef %tobool154, i1 noundef %tobool155, i1 noundef %tobool156, i1 noundef %tobool157, i1 noundef %tobool158, i8 noundef %82, i32 noundef %83)
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #4

; Function Attrs: noinline nounwind optnone
define internal void @tiled_matmul_outer(i64 noundef %dim_I, i64 noundef %dim_J, i64 noundef %dim_K, ptr noundef %A, ptr noundef %B, ptr noundef %D, ptr noundef %C, i64 noundef %stride_A, i64 noundef %stride_B, i64 noundef %stride_D, i64 noundef %stride_C, float noundef %A_scale_factor, float noundef %B_scale_factor, i32 noundef %D_scale_factor, i64 noundef %tile_I, i64 noundef %tile_J, i64 noundef %tile_K, i32 noundef %act, float noundef %scale, float noundef %bert_scale, i1 noundef %repeating_bias, i1 noundef %a_transpose, i1 noundef %b_transpose, i1 noundef %full_C, i1 noundef %low_D, i8 noundef %weightA, i32 noundef %dataflow) #0 {
entry:
  %dim_I.addr = alloca i64, align 8
  %dim_J.addr = alloca i64, align 8
  %dim_K.addr = alloca i64, align 8
  %A.addr = alloca ptr, align 8
  %B.addr = alloca ptr, align 8
  %D.addr = alloca ptr, align 8
  %C.addr = alloca ptr, align 8
  %stride_A.addr = alloca i64, align 8
  %stride_B.addr = alloca i64, align 8
  %stride_D.addr = alloca i64, align 8
  %stride_C.addr = alloca i64, align 8
  %A_scale_factor.addr = alloca float, align 4
  %B_scale_factor.addr = alloca float, align 4
  %D_scale_factor.addr = alloca i32, align 4
  %tile_I.addr = alloca i64, align 8
  %tile_J.addr = alloca i64, align 8
  %tile_K.addr = alloca i64, align 8
  %act.addr = alloca i32, align 4
  %scale.addr = alloca float, align 4
  %bert_scale.addr = alloca float, align 4
  %repeating_bias.addr = alloca i8, align 1
  %a_transpose.addr = alloca i8, align 1
  %b_transpose.addr = alloca i8, align 1
  %full_C.addr = alloca i8, align 1
  %low_D.addr = alloca i8, align 1
  %weightA.addr = alloca i8, align 1
  %dataflow.addr = alloca i32, align 4
  %dim_I_padded = alloca i64, align 8
  %dim_J_padded = alloca i64, align 8
  %dim_K_padded = alloca i64, align 8
  %I0 = alloca i64, align 8
  %J0 = alloca i64, align 8
  %K0 = alloca i64, align 8
  %last_I = alloca i64, align 8
  %last_J = alloca i64, align 8
  %last_K = alloca i64, align 8
  %padding_I = alloca i64, align 8
  %padding_J = alloca i64, align 8
  %padding_K = alloca i64, align 8
  %no_bias = alloca i8, align 1
  %sizeof_D = alloca i64, align 8
  %sizeof_C = alloca i64, align 8
  %inner = alloca ptr, align 8
  %i0 = alloca i64, align 8
  %j0 = alloca i64, align 8
  %k0 = alloca i64, align 8
  %pre = alloca ptr, align 8
  %bias_row = alloca i64, align 8
  %out = alloca ptr, align 8
  %I = alloca i64, align 8
  %J = alloca i64, align 8
  %K = alloca i64, align 8
  %pad_I = alloca i64, align 8
  %pad_J = alloca i64, align 8
  %pad_K = alloca i64, align 8
  %a = alloca ptr, align 8
  %b = alloca ptr, align 8
  store i64 %dim_I, ptr %dim_I.addr, align 8
  store i64 %dim_J, ptr %dim_J.addr, align 8
  store i64 %dim_K, ptr %dim_K.addr, align 8
  store ptr %A, ptr %A.addr, align 8
  store ptr %B, ptr %B.addr, align 8
  store ptr %D, ptr %D.addr, align 8
  store ptr %C, ptr %C.addr, align 8
  store i64 %stride_A, ptr %stride_A.addr, align 8
  store i64 %stride_B, ptr %stride_B.addr, align 8
  store i64 %stride_D, ptr %stride_D.addr, align 8
  store i64 %stride_C, ptr %stride_C.addr, align 8
  store float %A_scale_factor, ptr %A_scale_factor.addr, align 4
  store float %B_scale_factor, ptr %B_scale_factor.addr, align 4
  store i32 %D_scale_factor, ptr %D_scale_factor.addr, align 4
  store i64 %tile_I, ptr %tile_I.addr, align 8
  store i64 %tile_J, ptr %tile_J.addr, align 8
  store i64 %tile_K, ptr %tile_K.addr, align 8
  store i32 %act, ptr %act.addr, align 4
  store float %scale, ptr %scale.addr, align 4
  store float %bert_scale, ptr %bert_scale.addr, align 4
  %frombool = zext i1 %repeating_bias to i8
  store i8 %frombool, ptr %repeating_bias.addr, align 1
  %frombool1 = zext i1 %a_transpose to i8
  store i8 %frombool1, ptr %a_transpose.addr, align 1
  %frombool2 = zext i1 %b_transpose to i8
  store i8 %frombool2, ptr %b_transpose.addr, align 1
  %frombool3 = zext i1 %full_C to i8
  store i8 %frombool3, ptr %full_C.addr, align 1
  %frombool4 = zext i1 %low_D to i8
  store i8 %frombool4, ptr %low_D.addr, align 1
  store i8 %weightA, ptr %weightA.addr, align 1
  store i32 %dataflow, ptr %dataflow.addr, align 4
  %0 = load i64, ptr %dim_I.addr, align 8
  %div = udiv i64 %0, 16
  %1 = load i64, ptr %dim_I.addr, align 8
  %rem = urem i64 %1, 16
  %cmp = icmp ne i64 %rem, 0
  %conv = zext i1 %cmp to i32
  %conv5 = sext i32 %conv to i64
  %add = add i64 %div, %conv5
  %mul = mul i64 %add, 16
  store i64 %mul, ptr %dim_I_padded, align 8
  %2 = load i64, ptr %dim_J.addr, align 8
  %div6 = udiv i64 %2, 16
  %3 = load i64, ptr %dim_J.addr, align 8
  %rem7 = urem i64 %3, 16
  %cmp8 = icmp ne i64 %rem7, 0
  %conv9 = zext i1 %cmp8 to i32
  %conv10 = sext i32 %conv9 to i64
  %add11 = add i64 %div6, %conv10
  %mul12 = mul i64 %add11, 16
  store i64 %mul12, ptr %dim_J_padded, align 8
  %4 = load i64, ptr %dim_K.addr, align 8
  %div13 = udiv i64 %4, 16
  %5 = load i64, ptr %dim_K.addr, align 8
  %rem14 = urem i64 %5, 16
  %cmp15 = icmp ne i64 %rem14, 0
  %conv16 = zext i1 %cmp15 to i32
  %conv17 = sext i32 %conv16 to i64
  %add18 = add i64 %div13, %conv17
  %mul19 = mul i64 %add18, 16
  store i64 %mul19, ptr %dim_K_padded, align 8
  %6 = load i64, ptr %dim_I_padded, align 8
  %7 = load i64, ptr %tile_I.addr, align 8
  %mul20 = mul i64 %7, 16
  %div21 = udiv i64 %6, %mul20
  %8 = load i64, ptr %dim_I_padded, align 8
  %9 = load i64, ptr %tile_I.addr, align 8
  %mul22 = mul i64 %9, 16
  %rem23 = urem i64 %8, %mul22
  %cmp24 = icmp ne i64 %rem23, 0
  %conv25 = zext i1 %cmp24 to i32
  %conv26 = sext i32 %conv25 to i64
  %add27 = add i64 %div21, %conv26
  store i64 %add27, ptr %I0, align 8
  %10 = load i64, ptr %dim_J_padded, align 8
  %11 = load i64, ptr %tile_J.addr, align 8
  %mul28 = mul i64 %11, 16
  %div29 = udiv i64 %10, %mul28
  %12 = load i64, ptr %dim_J_padded, align 8
  %13 = load i64, ptr %tile_J.addr, align 8
  %mul30 = mul i64 %13, 16
  %rem31 = urem i64 %12, %mul30
  %cmp32 = icmp ne i64 %rem31, 0
  %conv33 = zext i1 %cmp32 to i32
  %conv34 = sext i32 %conv33 to i64
  %add35 = add i64 %div29, %conv34
  store i64 %add35, ptr %J0, align 8
  %14 = load i64, ptr %dim_K_padded, align 8
  %15 = load i64, ptr %tile_K.addr, align 8
  %mul36 = mul i64 %15, 16
  %div37 = udiv i64 %14, %mul36
  %16 = load i64, ptr %dim_K_padded, align 8
  %17 = load i64, ptr %tile_K.addr, align 8
  %mul38 = mul i64 %17, 16
  %rem39 = urem i64 %16, %mul38
  %cmp40 = icmp ne i64 %rem39, 0
  %conv41 = zext i1 %cmp40 to i32
  %conv42 = sext i32 %conv41 to i64
  %add43 = add i64 %div37, %conv42
  store i64 %add43, ptr %K0, align 8
  %18 = load i64, ptr %dim_I_padded, align 8
  %19 = load i64, ptr %tile_I.addr, align 8
  %mul44 = mul i64 %19, 16
  %rem45 = urem i64 %18, %mul44
  %cmp46 = icmp eq i64 %rem45, 0
  br i1 %cmp46, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  %20 = load i64, ptr %tile_I.addr, align 8
  br label %cond.end

cond.false:                                       ; preds = %entry
  %21 = load i64, ptr %dim_I_padded, align 8
  %div48 = udiv i64 %21, 16
  %22 = load i64, ptr %tile_I.addr, align 8
  %rem49 = urem i64 %div48, %22
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i64 [ %20, %cond.true ], [ %rem49, %cond.false ]
  store i64 %cond, ptr %last_I, align 8
  %23 = load i64, ptr %dim_J_padded, align 8
  %24 = load i64, ptr %tile_J.addr, align 8
  %mul50 = mul i64 %24, 16
  %rem51 = urem i64 %23, %mul50
  %cmp52 = icmp eq i64 %rem51, 0
  br i1 %cmp52, label %cond.true54, label %cond.false55

cond.true54:                                      ; preds = %cond.end
  %25 = load i64, ptr %tile_J.addr, align 8
  br label %cond.end58

cond.false55:                                     ; preds = %cond.end
  %26 = load i64, ptr %dim_J_padded, align 8
  %div56 = udiv i64 %26, 16
  %27 = load i64, ptr %tile_J.addr, align 8
  %rem57 = urem i64 %div56, %27
  br label %cond.end58

cond.end58:                                       ; preds = %cond.false55, %cond.true54
  %cond59 = phi i64 [ %25, %cond.true54 ], [ %rem57, %cond.false55 ]
  store i64 %cond59, ptr %last_J, align 8
  %28 = load i64, ptr %dim_K_padded, align 8
  %29 = load i64, ptr %tile_K.addr, align 8
  %mul60 = mul i64 %29, 16
  %rem61 = urem i64 %28, %mul60
  %cmp62 = icmp eq i64 %rem61, 0
  br i1 %cmp62, label %cond.true64, label %cond.false65

cond.true64:                                      ; preds = %cond.end58
  %30 = load i64, ptr %tile_K.addr, align 8
  br label %cond.end68

cond.false65:                                     ; preds = %cond.end58
  %31 = load i64, ptr %dim_K_padded, align 8
  %div66 = udiv i64 %31, 16
  %32 = load i64, ptr %tile_K.addr, align 8
  %rem67 = urem i64 %div66, %32
  br label %cond.end68

cond.end68:                                       ; preds = %cond.false65, %cond.true64
  %cond69 = phi i64 [ %30, %cond.true64 ], [ %rem67, %cond.false65 ]
  store i64 %cond69, ptr %last_K, align 8
  %33 = load i64, ptr %dim_I_padded, align 8
  %34 = load i64, ptr %dim_I.addr, align 8
  %sub = sub i64 %33, %34
  store i64 %sub, ptr %padding_I, align 8
  %35 = load i64, ptr %dim_J_padded, align 8
  %36 = load i64, ptr %dim_J.addr, align 8
  %sub70 = sub i64 %35, %36
  store i64 %sub70, ptr %padding_J, align 8
  %37 = load i64, ptr %dim_K_padded, align 8
  %38 = load i64, ptr %dim_K.addr, align 8
  %sub71 = sub i64 %37, %38
  store i64 %sub71, ptr %padding_K, align 8
  %39 = load ptr, ptr %D.addr, align 8
  %cmp72 = icmp eq ptr %39, null
  %frombool74 = zext i1 %cmp72 to i8
  store i8 %frombool74, ptr %no_bias, align 1
  %40 = load i8, ptr %no_bias, align 1
  %tobool = trunc i8 %40 to i1
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %cond.end68
  store ptr inttoptr (i64 1 to ptr), ptr %D.addr, align 8
  br label %if.end

if.end:                                           ; preds = %if.then, %cond.end68
  %41 = load i8, ptr %low_D.addr, align 1
  %tobool75 = trunc i8 %41 to i1
  %42 = zext i1 %tobool75 to i64
  %cond77 = select i1 %tobool75, i64 1, i64 4
  store i64 %cond77, ptr %sizeof_D, align 8
  %43 = load i8, ptr %full_C.addr, align 1
  %tobool78 = trunc i8 %43 to i1
  %44 = zext i1 %tobool78 to i64
  %cond80 = select i1 %tobool78, i64 4, i64 1
  store i64 %cond80, ptr %sizeof_C, align 8
  call void @llvm.riscv.configEx(i64 4575657221408489476, i64 281474976710656)
  %call = call signext i32 (ptr, ...) @printf(ptr noundef @.str.21)
  call void @llvm.riscv.configSt(i64 2, i64 4575657221408424000)
  %call81 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.22)
  call void @llvm.riscv.configLd(i64 4575657221409472769, i64 64)
  %call82 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.23)
  call void @llvm.riscv.configLd(i64 4575657221409472777, i64 64)
  %call83 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.23)
  call void @llvm.riscv.configLd(i64 4575657221409472785, i64 256)
  %call84 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.23)
  store ptr @sp_tiled_matmul_ws, ptr %inner, align 8
  store i64 0, ptr %i0, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.inc218, %if.end
  %45 = load i64, ptr %i0, align 8
  %46 = load i64, ptr %I0, align 8
  %cmp85 = icmp ult i64 %45, %46
  br i1 %cmp85, label %for.body, label %for.end220

for.body:                                         ; preds = %for.cond
  store i64 0, ptr %j0, align 8
  br label %for.cond87

for.cond87:                                       ; preds = %for.inc215, %for.body
  %47 = load i64, ptr %j0, align 8
  %48 = load i64, ptr %J0, align 8
  %cmp88 = icmp ult i64 %47, %48
  br i1 %cmp88, label %for.body90, label %for.end217

for.body90:                                       ; preds = %for.cond87
  store i64 0, ptr %k0, align 8
  br label %for.cond91

for.cond91:                                       ; preds = %for.inc, %for.body90
  %49 = load i64, ptr %k0, align 8
  %50 = load i64, ptr %K0, align 8
  %cmp92 = icmp ult i64 %49, %50
  br i1 %cmp92, label %for.body94, label %for.end

for.body94:                                       ; preds = %for.cond91
  %51 = load i64, ptr %k0, align 8
  %cmp95 = icmp ne i64 %51, 0
  br i1 %cmp95, label %if.then97, label %if.else

if.then97:                                        ; preds = %for.body94
  store ptr null, ptr %pre, align 8
  br label %if.end111

if.else:                                          ; preds = %for.body94
  %52 = load i8, ptr %repeating_bias.addr, align 1
  %tobool98 = trunc i8 %52 to i1
  br i1 %tobool98, label %cond.true100, label %cond.false101

cond.true100:                                     ; preds = %if.else
  br label %cond.end104

cond.false101:                                    ; preds = %if.else
  %53 = load i64, ptr %i0, align 8
  %54 = load i64, ptr %tile_I.addr, align 8
  %mul102 = mul i64 %53, %54
  %mul103 = mul i64 %mul102, 16
  br label %cond.end104

cond.end104:                                      ; preds = %cond.false101, %cond.true100
  %cond105 = phi i64 [ 0, %cond.true100 ], [ %mul103, %cond.false101 ]
  store i64 %cond105, ptr %bias_row, align 8
  %55 = load ptr, ptr %D.addr, align 8
  %56 = load i64, ptr %bias_row, align 8
  %57 = load i64, ptr %stride_D.addr, align 8
  %mul106 = mul i64 %56, %57
  %58 = load i64, ptr %j0, align 8
  %59 = load i64, ptr %tile_J.addr, align 8
  %mul107 = mul i64 %58, %59
  %mul108 = mul i64 %mul107, 16
  %add109 = add i64 %mul106, %mul108
  %60 = load i64, ptr %sizeof_D, align 8
  %mul110 = mul i64 %add109, %60
  %add.ptr = getelementptr inbounds i8, ptr %55, i64 %mul110
  store ptr %add.ptr, ptr %pre, align 8
  br label %if.end111

if.end111:                                        ; preds = %cond.end104, %if.then97
  %61 = load i64, ptr %k0, align 8
  %62 = load i64, ptr %K0, align 8
  %sub112 = sub i64 %62, 1
  %cmp113 = icmp eq i64 %61, %sub112
  br i1 %cmp113, label %cond.true115, label %cond.false124

cond.true115:                                     ; preds = %if.end111
  %63 = load ptr, ptr %C.addr, align 8
  %64 = load i64, ptr %i0, align 8
  %65 = load i64, ptr %tile_I.addr, align 8
  %mul116 = mul i64 %64, %65
  %mul117 = mul i64 %mul116, 16
  %66 = load i64, ptr %stride_C.addr, align 8
  %mul118 = mul i64 %mul117, %66
  %67 = load i64, ptr %j0, align 8
  %68 = load i64, ptr %tile_J.addr, align 8
  %mul119 = mul i64 %67, %68
  %mul120 = mul i64 %mul119, 16
  %add121 = add i64 %mul118, %mul120
  %69 = load i64, ptr %sizeof_C, align 8
  %mul122 = mul i64 %add121, %69
  %add.ptr123 = getelementptr inbounds i8, ptr %63, i64 %mul122
  br label %cond.end125

cond.false124:                                    ; preds = %if.end111
  br label %cond.end125

cond.end125:                                      ; preds = %cond.false124, %cond.true115
  %cond126 = phi ptr [ %add.ptr123, %cond.true115 ], [ null, %cond.false124 ]
  store ptr %cond126, ptr %out, align 8
  %70 = load i64, ptr %i0, align 8
  %71 = load i64, ptr %I0, align 8
  %sub127 = sub i64 %71, 1
  %cmp128 = icmp ult i64 %70, %sub127
  br i1 %cmp128, label %cond.true130, label %cond.false131

cond.true130:                                     ; preds = %cond.end125
  %72 = load i64, ptr %tile_I.addr, align 8
  br label %cond.end132

cond.false131:                                    ; preds = %cond.end125
  %73 = load i64, ptr %last_I, align 8
  br label %cond.end132

cond.end132:                                      ; preds = %cond.false131, %cond.true130
  %cond133 = phi i64 [ %72, %cond.true130 ], [ %73, %cond.false131 ]
  store i64 %cond133, ptr %I, align 8
  %74 = load i64, ptr %j0, align 8
  %75 = load i64, ptr %J0, align 8
  %sub134 = sub i64 %75, 1
  %cmp135 = icmp ult i64 %74, %sub134
  br i1 %cmp135, label %cond.true137, label %cond.false138

cond.true137:                                     ; preds = %cond.end132
  %76 = load i64, ptr %tile_J.addr, align 8
  br label %cond.end139

cond.false138:                                    ; preds = %cond.end132
  %77 = load i64, ptr %last_J, align 8
  br label %cond.end139

cond.end139:                                      ; preds = %cond.false138, %cond.true137
  %cond140 = phi i64 [ %76, %cond.true137 ], [ %77, %cond.false138 ]
  store i64 %cond140, ptr %J, align 8
  %78 = load i64, ptr %k0, align 8
  %79 = load i64, ptr %K0, align 8
  %sub141 = sub i64 %79, 1
  %cmp142 = icmp ult i64 %78, %sub141
  br i1 %cmp142, label %cond.true144, label %cond.false145

cond.true144:                                     ; preds = %cond.end139
  %80 = load i64, ptr %tile_K.addr, align 8
  br label %cond.end146

cond.false145:                                    ; preds = %cond.end139
  %81 = load i64, ptr %last_K, align 8
  br label %cond.end146

cond.end146:                                      ; preds = %cond.false145, %cond.true144
  %cond147 = phi i64 [ %80, %cond.true144 ], [ %81, %cond.false145 ]
  store i64 %cond147, ptr %K, align 8
  %82 = load i64, ptr %i0, align 8
  %83 = load i64, ptr %I0, align 8
  %sub148 = sub i64 %83, 1
  %cmp149 = icmp eq i64 %82, %sub148
  br i1 %cmp149, label %cond.true151, label %cond.false152

cond.true151:                                     ; preds = %cond.end146
  %84 = load i64, ptr %padding_I, align 8
  br label %cond.end153

cond.false152:                                    ; preds = %cond.end146
  br label %cond.end153

cond.end153:                                      ; preds = %cond.false152, %cond.true151
  %cond154 = phi i64 [ %84, %cond.true151 ], [ 0, %cond.false152 ]
  store i64 %cond154, ptr %pad_I, align 8
  %85 = load i64, ptr %j0, align 8
  %86 = load i64, ptr %J0, align 8
  %sub155 = sub i64 %86, 1
  %cmp156 = icmp eq i64 %85, %sub155
  br i1 %cmp156, label %cond.true158, label %cond.false159

cond.true158:                                     ; preds = %cond.end153
  %87 = load i64, ptr %padding_J, align 8
  br label %cond.end160

cond.false159:                                    ; preds = %cond.end153
  br label %cond.end160

cond.end160:                                      ; preds = %cond.false159, %cond.true158
  %cond161 = phi i64 [ %87, %cond.true158 ], [ 0, %cond.false159 ]
  store i64 %cond161, ptr %pad_J, align 8
  %88 = load i64, ptr %k0, align 8
  %89 = load i64, ptr %K0, align 8
  %sub162 = sub i64 %89, 1
  %cmp163 = icmp eq i64 %88, %sub162
  br i1 %cmp163, label %cond.true165, label %cond.false166

cond.true165:                                     ; preds = %cond.end160
  %90 = load i64, ptr %padding_K, align 8
  br label %cond.end167

cond.false166:                                    ; preds = %cond.end160
  br label %cond.end167

cond.end167:                                      ; preds = %cond.false166, %cond.true165
  %cond168 = phi i64 [ %90, %cond.true165 ], [ 0, %cond.false166 ]
  store i64 %cond168, ptr %pad_K, align 8
  %91 = load i8, ptr %a_transpose.addr, align 1
  %tobool169 = trunc i8 %91 to i1
  br i1 %tobool169, label %cond.true171, label %cond.false179

cond.true171:                                     ; preds = %cond.end167
  %92 = load ptr, ptr %A.addr, align 8
  %93 = load i64, ptr %k0, align 8
  %94 = load i64, ptr %tile_K.addr, align 8
  %mul172 = mul i64 %93, %94
  %mul173 = mul i64 %mul172, 16
  %95 = load i64, ptr %stride_A.addr, align 8
  %mul174 = mul i64 %mul173, %95
  %add.ptr175 = getelementptr inbounds i8, ptr %92, i64 %mul174
  %96 = load i64, ptr %i0, align 8
  %97 = load i64, ptr %tile_I.addr, align 8
  %mul176 = mul i64 %96, %97
  %mul177 = mul i64 %mul176, 16
  %add.ptr178 = getelementptr inbounds i8, ptr %add.ptr175, i64 %mul177
  br label %cond.end187

cond.false179:                                    ; preds = %cond.end167
  %98 = load ptr, ptr %A.addr, align 8
  %99 = load i64, ptr %i0, align 8
  %100 = load i64, ptr %tile_I.addr, align 8
  %mul180 = mul i64 %99, %100
  %mul181 = mul i64 %mul180, 16
  %101 = load i64, ptr %stride_A.addr, align 8
  %mul182 = mul i64 %mul181, %101
  %add.ptr183 = getelementptr inbounds i8, ptr %98, i64 %mul182
  %102 = load i64, ptr %k0, align 8
  %103 = load i64, ptr %tile_K.addr, align 8
  %mul184 = mul i64 %102, %103
  %mul185 = mul i64 %mul184, 16
  %add.ptr186 = getelementptr inbounds i8, ptr %add.ptr183, i64 %mul185
  br label %cond.end187

cond.end187:                                      ; preds = %cond.false179, %cond.true171
  %cond188 = phi ptr [ %add.ptr178, %cond.true171 ], [ %add.ptr186, %cond.false179 ]
  store ptr %cond188, ptr %a, align 8
  %104 = load i8, ptr %b_transpose.addr, align 1
  %tobool189 = trunc i8 %104 to i1
  br i1 %tobool189, label %cond.true191, label %cond.false199

cond.true191:                                     ; preds = %cond.end187
  %105 = load ptr, ptr %B.addr, align 8
  %106 = load i64, ptr %j0, align 8
  %107 = load i64, ptr %tile_J.addr, align 8
  %mul192 = mul i64 %106, %107
  %mul193 = mul i64 %mul192, 16
  %108 = load i64, ptr %stride_B.addr, align 8
  %mul194 = mul i64 %mul193, %108
  %add.ptr195 = getelementptr inbounds i8, ptr %105, i64 %mul194
  %109 = load i64, ptr %k0, align 8
  %110 = load i64, ptr %tile_K.addr, align 8
  %mul196 = mul i64 %109, %110
  %mul197 = mul i64 %mul196, 16
  %add.ptr198 = getelementptr inbounds i8, ptr %add.ptr195, i64 %mul197
  br label %cond.end207

cond.false199:                                    ; preds = %cond.end187
  %111 = load ptr, ptr %B.addr, align 8
  %112 = load i64, ptr %k0, align 8
  %113 = load i64, ptr %tile_K.addr, align 8
  %mul200 = mul i64 %112, %113
  %mul201 = mul i64 %mul200, 16
  %114 = load i64, ptr %stride_B.addr, align 8
  %mul202 = mul i64 %mul201, %114
  %add.ptr203 = getelementptr inbounds i8, ptr %111, i64 %mul202
  %115 = load i64, ptr %j0, align 8
  %116 = load i64, ptr %tile_J.addr, align 8
  %mul204 = mul i64 %115, %116
  %mul205 = mul i64 %mul204, 16
  %add.ptr206 = getelementptr inbounds i8, ptr %add.ptr203, i64 %mul205
  br label %cond.end207

cond.end207:                                      ; preds = %cond.false199, %cond.true191
  %cond208 = phi ptr [ %add.ptr198, %cond.true191 ], [ %add.ptr206, %cond.false199 ]
  store ptr %cond208, ptr %b, align 8
  %117 = load ptr, ptr %inner, align 8
  %118 = load ptr, ptr %a, align 8
  %119 = load ptr, ptr %b, align 8
  %120 = load ptr, ptr %pre, align 8
  %121 = load ptr, ptr %out, align 8
  %122 = load float, ptr %A_scale_factor.addr, align 4
  %123 = load float, ptr %B_scale_factor.addr, align 4
  %124 = load i32, ptr %D_scale_factor.addr, align 4
  %125 = load i64, ptr %I, align 8
  %126 = load i64, ptr %J, align 8
  %127 = load i64, ptr %K, align 8
  %128 = load i64, ptr %pad_I, align 8
  %129 = load i64, ptr %pad_J, align 8
  %130 = load i64, ptr %pad_K, align 8
  %131 = load i64, ptr %stride_A.addr, align 8
  %132 = load i64, ptr %stride_B.addr, align 8
  %133 = load i64, ptr %stride_D.addr, align 8
  %134 = load i64, ptr %stride_C.addr, align 8
  %135 = load i8, ptr %a_transpose.addr, align 1
  %tobool209 = trunc i8 %135 to i1
  %136 = load i8, ptr %b_transpose.addr, align 1
  %tobool210 = trunc i8 %136 to i1
  %137 = load i8, ptr %full_C.addr, align 1
  %tobool211 = trunc i8 %137 to i1
  %138 = load i8, ptr %low_D.addr, align 1
  %tobool212 = trunc i8 %138 to i1
  %139 = load i8, ptr %no_bias, align 1
  %tobool213 = trunc i8 %139 to i1
  %140 = load i8, ptr %repeating_bias.addr, align 1
  %tobool214 = trunc i8 %140 to i1
  %141 = load i32, ptr %act.addr, align 4
  call void %117(ptr noundef %118, ptr noundef %119, ptr noundef %120, ptr noundef %121, float noundef %122, float noundef %123, i32 noundef signext %124, i64 noundef %125, i64 noundef %126, i64 noundef %127, i64 noundef %128, i64 noundef %129, i64 noundef %130, i64 noundef %131, i64 noundef %132, i64 noundef %133, i64 noundef %134, i1 noundef %tobool209, i1 noundef %tobool210, i1 noundef %tobool211, i1 noundef %tobool212, i1 noundef %tobool213, i1 noundef %tobool214, i32 noundef %141)
  br label %for.inc

for.inc:                                          ; preds = %cond.end207
  %142 = load i64, ptr %k0, align 8
  %inc = add i64 %142, 1
  store i64 %inc, ptr %k0, align 8
  br label %for.cond91, !llvm.loop !22

for.end:                                          ; preds = %for.cond91
  br label %for.inc215

for.inc215:                                       ; preds = %for.end
  %143 = load i64, ptr %j0, align 8
  %inc216 = add i64 %143, 1
  store i64 %inc216, ptr %j0, align 8
  br label %for.cond87, !llvm.loop !23

for.end217:                                       ; preds = %for.cond87
  br label %for.inc218

for.inc218:                                       ; preds = %for.end217
  %144 = load i64, ptr %i0, align 8
  %inc219 = add i64 %144, 1
  store i64 %inc219, ptr %i0, align 8
  br label %for.cond, !llvm.loop !24

for.end220:                                       ; preds = %for.cond
  ret void
}

; Function Attrs: noinline nounwind optnone
define internal void @sp_tiled_matmul_ws(ptr noundef %A, ptr noundef %B, ptr noundef %D, ptr noundef %C, float noundef %A_scale_factor, float noundef %B_scale_factor, i32 noundef signext %D_scale_factor, i64 noundef %I, i64 noundef %J, i64 noundef %K, i64 noundef %pad_I, i64 noundef %pad_J, i64 noundef %pad_K, i64 noundef %A_row_stride, i64 noundef %B_row_stride, i64 noundef %D_row_stride, i64 noundef %C_row_stride, i1 noundef %a_transpose, i1 noundef %b_transpose, i1 noundef %full_C, i1 noundef %low_D, i1 noundef %no_bias, i1 noundef %repeating_bias, i32 noundef %act) #0 {
entry:
  %A.addr = alloca ptr, align 8
  %B.addr = alloca ptr, align 8
  %D.addr = alloca ptr, align 8
  %C.addr = alloca ptr, align 8
  %A_scale_factor.addr = alloca float, align 4
  %B_scale_factor.addr = alloca float, align 4
  %D_scale_factor.addr = alloca i32, align 4
  %I.addr = alloca i64, align 8
  %J.addr = alloca i64, align 8
  %K.addr = alloca i64, align 8
  %pad_I.addr = alloca i64, align 8
  %pad_J.addr = alloca i64, align 8
  %pad_K.addr = alloca i64, align 8
  %A_row_stride.addr = alloca i64, align 8
  %B_row_stride.addr = alloca i64, align 8
  %D_row_stride.addr = alloca i64, align 8
  %C_row_stride.addr = alloca i64, align 8
  %a_transpose.addr = alloca i8, align 1
  %b_transpose.addr = alloca i8, align 1
  %full_C.addr = alloca i8, align 1
  %low_D.addr = alloca i8, align 1
  %no_bias.addr = alloca i8, align 1
  %repeating_bias.addr = alloca i8, align 1
  %act.addr = alloca i32, align 4
  %Aaddr = alloca i64, align 8
  %Baddr = alloca i64, align 8
  %Daddr = alloca i64, align 8
  %Caddr = alloca i64, align 8
  store ptr %A, ptr %A.addr, align 8
  store ptr %B, ptr %B.addr, align 8
  store ptr %D, ptr %D.addr, align 8
  store ptr %C, ptr %C.addr, align 8
  store float %A_scale_factor, ptr %A_scale_factor.addr, align 4
  store float %B_scale_factor, ptr %B_scale_factor.addr, align 4
  store i32 %D_scale_factor, ptr %D_scale_factor.addr, align 4
  store i64 %I, ptr %I.addr, align 8
  store i64 %J, ptr %J.addr, align 8
  store i64 %K, ptr %K.addr, align 8
  store i64 %pad_I, ptr %pad_I.addr, align 8
  store i64 %pad_J, ptr %pad_J.addr, align 8
  store i64 %pad_K, ptr %pad_K.addr, align 8
  store i64 %A_row_stride, ptr %A_row_stride.addr, align 8
  store i64 %B_row_stride, ptr %B_row_stride.addr, align 8
  store i64 %D_row_stride, ptr %D_row_stride.addr, align 8
  store i64 %C_row_stride, ptr %C_row_stride.addr, align 8
  %frombool = zext i1 %a_transpose to i8
  store i8 %frombool, ptr %a_transpose.addr, align 1
  %frombool1 = zext i1 %b_transpose to i8
  store i8 %frombool1, ptr %b_transpose.addr, align 1
  %frombool2 = zext i1 %full_C to i8
  store i8 %frombool2, ptr %full_C.addr, align 1
  %frombool3 = zext i1 %low_D to i8
  store i8 %frombool3, ptr %low_D.addr, align 1
  %frombool4 = zext i1 %no_bias to i8
  store i8 %frombool4, ptr %no_bias.addr, align 1
  %frombool5 = zext i1 %repeating_bias to i8
  store i8 %frombool5, ptr %repeating_bias.addr, align 1
  store i32 %act, ptr %act.addr, align 4
  %0 = load ptr, ptr %A.addr, align 8
  %1 = ptrtoint ptr %0 to i64
  store i64 %1, ptr %Aaddr, align 8
  %2 = load ptr, ptr %B.addr, align 8
  %3 = ptrtoint ptr %2 to i64
  store i64 %3, ptr %Baddr, align 8
  %4 = load ptr, ptr %D.addr, align 8
  %5 = ptrtoint ptr %4 to i64
  store i64 %5, ptr %Daddr, align 8
  %6 = load ptr, ptr %C.addr, align 8
  %7 = ptrtoint ptr %6 to i64
  store i64 %7, ptr %Caddr, align 8
  call void @llvm.riscv.loopWsConfigBounds(i64 0, i64 17180131332)
  %call = call signext i32 (ptr, ...) @printf(ptr noundef @.str.24)
  call void @llvm.riscv.loopWsConfigAddrsAB(i64 %1, i64 %3);
  %call6 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.25)
  call void @llvm.riscv.loopWsConfigAddrsDC(i64 0, i64 %7)
  %call7 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.26)
  call void @llvm.riscv.loopWsConfigStridesAB(i64 64, i64 64)
  %call8 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.27)
  call void @llvm.riscv.loopWsConfigStridesDC(i64 64, i64 64)
  %call9 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.28)
  call void @llvm.riscv.loopWs(i64 0, i64 0)
  %call10 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.29)
  ret void
}

attributes #0 = { noinline nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+m,+relax,-save-restore" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+m,+relax,-save-restore" }
attributes #2 = { noreturn "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+m,+relax,-save-restore" }
attributes #3 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+m,+relax,-save-restore" }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { noreturn }
attributes #6 = { nounwind }

declare void @llvm.riscv.configEx(i64, i64)
declare void @llvm.riscv.configLd(i64, i64)
declare void @llvm.riscv.configSt(i64, i64)
declare void @llvm.riscv.loopWsConfigBounds(i64, i64)
declare void @llvm.riscv.loopWsConfigAddrsAB(i64, i64)
declare void @llvm.riscv.loopWsConfigAddrsDC(i64, i64)
declare void @llvm.riscv.loopWsConfigStridesAB(i64, i64)
declare void @llvm.riscv.loopWsConfigStridesDC(i64, i64)
declare void @llvm.riscv.loopWs(i64, i64)

!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.mustprogress"}
!7 = distinct !{!7, !6}
!8 = distinct !{!8, !6}
!9 = distinct !{!9, !6}
!10 = distinct !{!10, !6}
!11 = distinct !{!11, !6}
!12 = distinct !{!12, !6}
!13 = distinct !{!13, !6}
!14 = distinct !{!14, !6}
!15 = distinct !{!15, !6}
!16 = distinct !{!16, !6}
!17 = distinct !{!17, !6}
!18 = distinct !{!18, !6}
!19 = distinct !{!19, !6}
!20 = distinct !{!20, !6}
!21 = !{i64 2479}
!22 = distinct !{!22, !6}
!23 = distinct !{!23, !6}
!24 = distinct !{!24, !6}
