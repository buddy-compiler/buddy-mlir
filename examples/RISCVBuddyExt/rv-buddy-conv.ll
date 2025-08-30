@.str = private unnamed_addr constant [29 x i8] c"conv out_dim is not correct\0A\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"%d \00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@.str.3 = private unnamed_addr constant [20 x i8] c"Randomize input...\0A\00", align 1
@.str.4 = private unnamed_addr constant [22 x i8] c"Randomize weights...\0A\00", align 1
@.str.5 = private unnamed_addr constant [19 x i8] c"Randomize bias...\0A\00", align 1
@.str.6 = private unnamed_addr constant [13 x i8] c"CPU conv...\0A\00", align 1
@.str.7 = private unnamed_addr constant [27 x i8] c"CPU conv took %llu cycles\0A\00", align 1
@main.weights_mat = internal global [162 x [19 x i8]] zeroinitializer, align 1
@main.output_mat = internal global [162 x [19 x i8]] zeroinitializer, align 1
@.str.8 = private unnamed_addr constant [20 x i8] c"Flatten weights...\0A\00", align 1
@.str.9 = private unnamed_addr constant [17 x i8] c"Gemmini conv...\0A\00", align 1
@.str.10 = private unnamed_addr constant [31 x i8] c"Gemmini conv took %llu cycles\0A\00", align 1
@.str.11 = private unnamed_addr constant [16 x i8] c"printf output.\0A\00", align 1
@.str.12 = private unnamed_addr constant [20 x i8] c"printf output_mat.\0A\00", align 1
@.str.13 = private unnamed_addr constant [17 x i8] c"tile_conv_auto.\0A\00", align 1
@.str.14 = private unnamed_addr constant [146 x i8] c"batch_size:%d in_dim:%d in_channels:%d, out_channels %d, out_dim:%d, stride:%d, input_dilation:%d, kernel_dilation:%d, padding:%d, kernel_dim:%d\0A\00", align 1
@.str.15 = private unnamed_addr constant [220 x i8] c"batch_size:%d, in_dim:%d, in_channels:%d, out_channels:%d, out_dim:%d, stride:%d, input_dilation:%d, kernel_dilation:%d, padding:%d, kernel_dim:%d, batchs:%d, porows:%d, pocols:%d, pochs:%d, krows:%d, kcols:%d, kchs:%d\0A\00", align 1
@.str.16 = private unnamed_addr constant [12 x i8] c"config_st.\0A\00", align 1
@.str.17 = private unnamed_addr constant [12 x i8] c"config_ex.\0A\00", align 1
@sp_tiled_conv.D_sp_addr_row = internal global i32 0, align 4
@sp_tiled_conv.C_sp_addr_row = internal global i32 0, align 4
@.str.18 = private unnamed_addr constant [23 x i8] c"loop_conv_ws_config1.\0A\00", align 1
@.str.19 = private unnamed_addr constant [23 x i8] c"loop_conv_ws_config2.\0A\00", align 1
@.str.20 = private unnamed_addr constant [23 x i8] c"loop_conv_ws_config3.\0A\00", align 1
@.str.21 = private unnamed_addr constant [23 x i8] c"loop_conv_ws_config4.\0A\00", align 1
@.str.22 = private unnamed_addr constant [23 x i8] c"loop_conv_ws_config5.\0A\00", align 1
@.str.23 = private unnamed_addr constant [23 x i8] c"loop_conv_ws_config6.\0A\00", align 1
@.str.24 = private unnamed_addr constant [15 x i8] c"loop_conv_ws.\0A\00", align 1

; Function Attrs: noinline nounwind optnone
define dso_local void @init_random(ptr noundef %buf, i32 noundef signext %len) #0 {
entry:
  %buf.addr = alloca ptr, align 8
  %len.addr = alloca i32, align 4
  %ptr = alloca ptr, align 8
  store ptr %buf, ptr %buf.addr, align 8
  store i32 %len, ptr %len.addr, align 4
  %0 = load ptr, ptr %buf.addr, align 8
  store ptr %0, ptr %ptr, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %1 = load ptr, ptr %ptr, align 8
  %2 = load ptr, ptr %buf.addr, align 8
  %3 = load i32, ptr %len.addr, align 4
  %idx.ext = sext i32 %3 to i64
  %add.ptr = getelementptr inbounds i8, ptr %2, i64 %idx.ext
  %cmp = icmp ult ptr %1, %add.ptr
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %call = call signext i32 @rand()
  %rem = srem i32 %call, 5
  %sub = sub nsw i32 %rem, 2
  %conv = trunc i32 %sub to i8
  %4 = load ptr, ptr %ptr, align 8
  store i8 %conv, ptr %4, align 1
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load ptr, ptr %ptr, align 8
  %incdec.ptr = getelementptr inbounds i8, ptr %5, i32 1
  store ptr %incdec.ptr, ptr %ptr, align 8
  br label %for.cond, !llvm.loop !5

for.end:                                          ; preds = %for.cond
  ret void
}

declare dso_local signext i32 @rand() #1

; Function Attrs: noinline nounwind optnone
define dso_local void @init_random_acc(ptr noundef %buf, i32 noundef signext %len) #0 {
entry:
  %buf.addr = alloca ptr, align 8
  %len.addr = alloca i32, align 4
  %ptr = alloca ptr, align 8
  store ptr %buf, ptr %buf.addr, align 8
  store i32 %len, ptr %len.addr, align 4
  %0 = load ptr, ptr %buf.addr, align 8
  store ptr %0, ptr %ptr, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %1 = load ptr, ptr %ptr, align 8
  %2 = load ptr, ptr %buf.addr, align 8
  %3 = load i32, ptr %len.addr, align 4
  %idx.ext = sext i32 %3 to i64
  %add.ptr = getelementptr inbounds i32, ptr %2, i64 %idx.ext
  %cmp = icmp ult ptr %1, %add.ptr
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %call = call signext i32 @rand()
  %rem = srem i32 %call, 5
  %sub = sub nsw i32 %rem, 2
  %4 = load ptr, ptr %ptr, align 8
  store i32 %sub, ptr %4, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load ptr, ptr %ptr, align 8
  %incdec.ptr = getelementptr inbounds i32, ptr %5, i32 1
  store ptr %incdec.ptr, ptr %ptr, align 8
  br label %for.cond, !llvm.loop !7

for.end:                                          ; preds = %for.cond
  ret void
}

; Function Attrs: noinline nounwind optnone
define dso_local zeroext i1 @vec_is_equal(ptr noundef %a, ptr noundef %b, i32 noundef signext %len) #0 {
entry:
  %retval = alloca i1, align 1
  %a.addr = alloca ptr, align 8
  %b.addr = alloca ptr, align 8
  %len.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store ptr %a, ptr %a.addr, align 8
  store ptr %b, ptr %b.addr, align 8
  store i32 %len, ptr %len.addr, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %1 = load i32, ptr %len.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %2 = load ptr, ptr %a.addr, align 8
  %3 = load i32, ptr %i, align 4
  %idxprom = sext i32 %3 to i64
  %arrayidx = getelementptr inbounds i8, ptr %2, i64 %idxprom
  %4 = load i8, ptr %arrayidx, align 1
  %conv = sext i8 %4 to i32
  %5 = load ptr, ptr %b.addr, align 8
  %6 = load i32, ptr %i, align 4
  %idxprom1 = sext i32 %6 to i64
  %arrayidx2 = getelementptr inbounds i8, ptr %5, i64 %idxprom1
  %7 = load i8, ptr %arrayidx2, align 1
  %conv3 = sext i8 %7 to i32
  %cmp4 = icmp ne i32 %conv, %conv3
  br i1 %cmp4, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  store i1 false, ptr %retval, align 1
  br label %return

if.end:                                           ; preds = %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %8 = load i32, ptr %i, align 4
  %inc = add nsw i32 %8, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond, !llvm.loop !8

for.end:                                          ; preds = %for.cond
  store i1 true, ptr %retval, align 1
  br label %return

return:                                           ; preds = %for.end, %if.then
  %9 = load i1, ptr %retval, align 1
  ret i1 %9
}

; Function Attrs: noinline nounwind optnone
define dso_local void @conv(i32 noundef signext %batch_size, i32 noundef signext %in_channels, i32 noundef signext %in_dim, i32 noundef signext %out_channels, i32 noundef signext %kernel_dim, i32 noundef signext %out_dim, i32 noundef signext %stride, i32 noundef signext %padding, ptr noundef %input, ptr noundef %weights, ptr noundef %bias, ptr noundef %output) #0 {
entry:
  %batch_size.addr = alloca i32, align 4
  %in_channels.addr = alloca i32, align 4
  %in_dim.addr = alloca i32, align 4
  %out_channels.addr = alloca i32, align 4
  %kernel_dim.addr = alloca i32, align 4
  %out_dim.addr = alloca i32, align 4
  %stride.addr = alloca i32, align 4
  %padding.addr = alloca i32, align 4
  %input.addr = alloca ptr, align 8
  %weights.addr = alloca ptr, align 8
  %bias.addr = alloca ptr, align 8
  %output.addr = alloca ptr, align 8
  %b = alloca i32, align 4
  %orow = alloca i32, align 4
  %ocol = alloca i32, align 4
  %och = alloca i32, align 4
  %result = alloca i32, align 4
  %krow = alloca i32, align 4
  %kcol = alloca i32, align 4
  %kch = alloca i32, align 4
  %irow = alloca i32, align 4
  %icol = alloca i32, align 4
  %pixel = alloca i8, align 1
  store i32 %batch_size, ptr %batch_size.addr, align 4
  store i32 %in_channels, ptr %in_channels.addr, align 4
  store i32 %in_dim, ptr %in_dim.addr, align 4
  store i32 %out_channels, ptr %out_channels.addr, align 4
  store i32 %kernel_dim, ptr %kernel_dim.addr, align 4
  store i32 %out_dim, ptr %out_dim.addr, align 4
  store i32 %stride, ptr %stride.addr, align 4
  store i32 %padding, ptr %padding.addr, align 4
  store ptr %input, ptr %input.addr, align 8
  store ptr %weights, ptr %weights.addr, align 8
  store ptr %bias, ptr %bias.addr, align 8
  store ptr %output, ptr %output.addr, align 8
  %0 = load i32, ptr %batch_size.addr, align 4
  %1 = zext i32 %0 to i64
  %2 = load i32, ptr %in_dim.addr, align 4
  %3 = zext i32 %2 to i64
  %4 = load i32, ptr %in_dim.addr, align 4
  %5 = zext i32 %4 to i64
  %6 = load i32, ptr %in_channels.addr, align 4
  %7 = zext i32 %6 to i64
  %8 = load i32, ptr %out_channels.addr, align 4
  %9 = zext i32 %8 to i64
  %10 = load i32, ptr %kernel_dim.addr, align 4
  %11 = zext i32 %10 to i64
  %12 = load i32, ptr %kernel_dim.addr, align 4
  %13 = zext i32 %12 to i64
  %14 = load i32, ptr %in_channels.addr, align 4
  %15 = zext i32 %14 to i64
  %16 = load i32, ptr %out_channels.addr, align 4
  %17 = zext i32 %16 to i64
  %18 = load i32, ptr %batch_size.addr, align 4
  %19 = zext i32 %18 to i64
  %20 = load i32, ptr %out_dim.addr, align 4
  %21 = zext i32 %20 to i64
  %22 = load i32, ptr %out_dim.addr, align 4
  %23 = zext i32 %22 to i64
  %24 = load i32, ptr %out_channels.addr, align 4
  %25 = zext i32 %24 to i64
  %26 = load i32, ptr %out_dim.addr, align 4
  %27 = load i32, ptr %in_dim.addr, align 4
  %28 = load i32, ptr %padding.addr, align 4
  %mul = mul nsw i32 2, %28
  %add = add nsw i32 %27, %mul
  %29 = load i32, ptr %kernel_dim.addr, align 4
  %sub = sub nsw i32 %add, %29
  %30 = load i32, ptr %stride.addr, align 4
  %div = sdiv i32 %sub, %30
  %add1 = add nsw i32 %div, 1
  %cmp = icmp ne i32 %26, %add1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call = call signext i32 (ptr, ...) @printf(ptr noundef @.str)
  call void @exit(i32 noundef signext 1) #3
  unreachable

if.end:                                           ; preds = %entry
  store i32 0, ptr %b, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc90, %if.end
  %31 = load i32, ptr %b, align 4
  %32 = load i32, ptr %batch_size.addr, align 4
  %cmp2 = icmp slt i32 %31, %32
  br i1 %cmp2, label %for.body, label %for.end92

for.body:                                         ; preds = %for.cond
  store i32 0, ptr %orow, align 4
  br label %for.cond3

for.cond3:                                        ; preds = %for.inc87, %for.body
  %33 = load i32, ptr %orow, align 4
  %34 = load i32, ptr %out_dim.addr, align 4
  %cmp4 = icmp slt i32 %33, %34
  br i1 %cmp4, label %for.body5, label %for.end89

for.body5:                                        ; preds = %for.cond3
  store i32 0, ptr %ocol, align 4
  br label %for.cond6

for.cond6:                                        ; preds = %for.inc84, %for.body5
  %35 = load i32, ptr %ocol, align 4
  %36 = load i32, ptr %out_dim.addr, align 4
  %cmp7 = icmp slt i32 %35, %36
  br i1 %cmp7, label %for.body8, label %for.end86

for.body8:                                        ; preds = %for.cond6
  store i32 0, ptr %och, align 4
  br label %for.cond9

for.cond9:                                        ; preds = %for.inc81, %for.body8
  %37 = load i32, ptr %och, align 4
  %38 = load i32, ptr %out_channels.addr, align 4
  %cmp10 = icmp slt i32 %37, %38
  br i1 %cmp10, label %for.body11, label %for.end83

for.body11:                                       ; preds = %for.cond9
  %39 = load ptr, ptr %bias.addr, align 8
  %40 = load i32, ptr %och, align 4
  %idxprom = sext i32 %40 to i64
  %arrayidx = getelementptr inbounds i32, ptr %39, i64 %idxprom
  %41 = load i32, ptr %arrayidx, align 4
  store i32 %41, ptr %result, align 4
  store i32 0, ptr %krow, align 4
  br label %for.cond12

for.cond12:                                       ; preds = %for.inc57, %for.body11
  %42 = load i32, ptr %krow, align 4
  %43 = load i32, ptr %kernel_dim.addr, align 4
  %cmp13 = icmp slt i32 %42, %43
  br i1 %cmp13, label %for.body14, label %for.end59

for.body14:                                       ; preds = %for.cond12
  store i32 0, ptr %kcol, align 4
  br label %for.cond15

for.cond15:                                       ; preds = %for.inc54, %for.body14
  %44 = load i32, ptr %kcol, align 4
  %45 = load i32, ptr %kernel_dim.addr, align 4
  %cmp16 = icmp slt i32 %44, %45
  br i1 %cmp16, label %for.body17, label %for.end56

for.body17:                                       ; preds = %for.cond15
  store i32 0, ptr %kch, align 4
  br label %for.cond18

for.cond18:                                       ; preds = %for.inc, %for.body17
  %46 = load i32, ptr %kch, align 4
  %47 = load i32, ptr %in_channels.addr, align 4
  %cmp19 = icmp slt i32 %46, %47
  br i1 %cmp19, label %for.body20, label %for.end

for.body20:                                       ; preds = %for.cond18
  %48 = load i32, ptr %orow, align 4
  %49 = load i32, ptr %stride.addr, align 4
  %mul21 = mul nsw i32 %48, %49
  %50 = load i32, ptr %krow, align 4
  %add22 = add nsw i32 %mul21, %50
  %51 = load i32, ptr %padding.addr, align 4
  %sub23 = sub nsw i32 %add22, %51
  store i32 %sub23, ptr %irow, align 4
  %52 = load i32, ptr %ocol, align 4
  %53 = load i32, ptr %stride.addr, align 4
  %mul24 = mul nsw i32 %52, %53
  %54 = load i32, ptr %kcol, align 4
  %add25 = add nsw i32 %mul24, %54
  %55 = load i32, ptr %padding.addr, align 4
  %sub26 = sub nsw i32 %add25, %55
  store i32 %sub26, ptr %icol, align 4
  %56 = load i32, ptr %irow, align 4
  %cmp27 = icmp slt i32 %56, 0
  br i1 %cmp27, label %cond.true, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %for.body20
  %57 = load i32, ptr %irow, align 4
  %58 = load i32, ptr %in_dim.addr, align 4
  %cmp28 = icmp sge i32 %57, %58
  br i1 %cmp28, label %cond.true, label %lor.lhs.false29

lor.lhs.false29:                                  ; preds = %lor.lhs.false
  %59 = load i32, ptr %icol, align 4
  %cmp30 = icmp slt i32 %59, 0
  br i1 %cmp30, label %cond.true, label %lor.lhs.false31

lor.lhs.false31:                                  ; preds = %lor.lhs.false29
  %60 = load i32, ptr %icol, align 4
  %61 = load i32, ptr %in_dim.addr, align 4
  %cmp32 = icmp sge i32 %60, %61
  br i1 %cmp32, label %cond.true, label %cond.false

cond.true:                                        ; preds = %lor.lhs.false31, %lor.lhs.false29, %lor.lhs.false, %for.body20
  br label %cond.end

cond.false:                                       ; preds = %lor.lhs.false31
  %62 = load ptr, ptr %input.addr, align 8
  %63 = load i32, ptr %b, align 4
  %idxprom33 = sext i32 %63 to i64
  %64 = mul nuw i64 %3, %5
  %65 = mul nuw i64 %64, %7
  %66 = mul nsw i64 %idxprom33, %65
  %arrayidx34 = getelementptr inbounds i8, ptr %62, i64 %66
  %67 = load i32, ptr %irow, align 4
  %idxprom35 = sext i32 %67 to i64
  %68 = mul nuw i64 %5, %7
  %69 = mul nsw i64 %idxprom35, %68
  %arrayidx36 = getelementptr inbounds i8, ptr %arrayidx34, i64 %69
  %70 = load i32, ptr %icol, align 4
  %idxprom37 = sext i32 %70 to i64
  %71 = mul nsw i64 %idxprom37, %7
  %arrayidx38 = getelementptr inbounds i8, ptr %arrayidx36, i64 %71
  %72 = load i32, ptr %kch, align 4
  %idxprom39 = sext i32 %72 to i64
  %arrayidx40 = getelementptr inbounds i8, ptr %arrayidx38, i64 %idxprom39
  %73 = load i8, ptr %arrayidx40, align 1
  %conv = sext i8 %73 to i32
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ 0, %cond.true ], [ %conv, %cond.false ]
  %conv41 = trunc i32 %cond to i8
  store i8 %conv41, ptr %pixel, align 1
  %74 = load ptr, ptr %weights.addr, align 8
  %75 = load i32, ptr %och, align 4
  %idxprom42 = sext i32 %75 to i64
  %76 = mul nuw i64 %11, %13
  %77 = mul nuw i64 %76, %15
  %78 = mul nsw i64 %idxprom42, %77
  %arrayidx43 = getelementptr inbounds i8, ptr %74, i64 %78
  %79 = load i32, ptr %krow, align 4
  %idxprom44 = sext i32 %79 to i64
  %80 = mul nuw i64 %13, %15
  %81 = mul nsw i64 %idxprom44, %80
  %arrayidx45 = getelementptr inbounds i8, ptr %arrayidx43, i64 %81
  %82 = load i32, ptr %kcol, align 4
  %idxprom46 = sext i32 %82 to i64
  %83 = mul nsw i64 %idxprom46, %15
  %arrayidx47 = getelementptr inbounds i8, ptr %arrayidx45, i64 %83
  %84 = load i32, ptr %kch, align 4
  %idxprom48 = sext i32 %84 to i64
  %arrayidx49 = getelementptr inbounds i8, ptr %arrayidx47, i64 %idxprom48
  %85 = load i8, ptr %arrayidx49, align 1
  %conv50 = sext i8 %85 to i32
  %86 = load i8, ptr %pixel, align 1
  %conv51 = sext i8 %86 to i32
  %mul52 = mul nsw i32 %conv50, %conv51
  %87 = load i32, ptr %result, align 4
  %add53 = add nsw i32 %87, %mul52
  store i32 %add53, ptr %result, align 4
  br label %for.inc

for.inc:                                          ; preds = %cond.end
  %88 = load i32, ptr %kch, align 4
  %inc = add nsw i32 %88, 1
  store i32 %inc, ptr %kch, align 4
  br label %for.cond18, !llvm.loop !9

for.end:                                          ; preds = %for.cond18
  br label %for.inc54

for.inc54:                                        ; preds = %for.end
  %89 = load i32, ptr %kcol, align 4
  %inc55 = add nsw i32 %89, 1
  store i32 %inc55, ptr %kcol, align 4
  br label %for.cond15, !llvm.loop !10

for.end56:                                        ; preds = %for.cond15
  br label %for.inc57

for.inc57:                                        ; preds = %for.end56
  %90 = load i32, ptr %krow, align 4
  %inc58 = add nsw i32 %90, 1
  store i32 %inc58, ptr %krow, align 4
  br label %for.cond12, !llvm.loop !11

for.end59:                                        ; preds = %for.cond12
  %91 = load i32, ptr %result, align 4
  %cmp60 = icmp sgt i32 %91, 127
  br i1 %cmp60, label %cond.true62, label %cond.false63

cond.true62:                                      ; preds = %for.end59
  br label %cond.end70

cond.false63:                                     ; preds = %for.end59
  %92 = load i32, ptr %result, align 4
  %cmp64 = icmp slt i32 %92, -128
  br i1 %cmp64, label %cond.true66, label %cond.false67

cond.true66:                                      ; preds = %cond.false63
  br label %cond.end68

cond.false67:                                     ; preds = %cond.false63
  %93 = load i32, ptr %result, align 4
  br label %cond.end68

cond.end68:                                       ; preds = %cond.false67, %cond.true66
  %cond69 = phi i32 [ -128, %cond.true66 ], [ %93, %cond.false67 ]
  br label %cond.end70

cond.end70:                                       ; preds = %cond.end68, %cond.true62
  %cond71 = phi i32 [ 127, %cond.true62 ], [ %cond69, %cond.end68 ]
  store i32 %cond71, ptr %result, align 4
  %94 = load i32, ptr %result, align 4
  %conv72 = trunc i32 %94 to i8
  %95 = load ptr, ptr %output.addr, align 8
  %96 = load i32, ptr %b, align 4
  %idxprom73 = sext i32 %96 to i64
  %97 = mul nuw i64 %21, %23
  %98 = mul nuw i64 %97, %25
  %99 = mul nsw i64 %idxprom73, %98
  %arrayidx74 = getelementptr inbounds i8, ptr %95, i64 %99
  %100 = load i32, ptr %orow, align 4
  %idxprom75 = sext i32 %100 to i64
  %101 = mul nuw i64 %23, %25
  %102 = mul nsw i64 %idxprom75, %101
  %arrayidx76 = getelementptr inbounds i8, ptr %arrayidx74, i64 %102
  %103 = load i32, ptr %ocol, align 4
  %idxprom77 = sext i32 %103 to i64
  %104 = mul nsw i64 %idxprom77, %25
  %arrayidx78 = getelementptr inbounds i8, ptr %arrayidx76, i64 %104
  %105 = load i32, ptr %och, align 4
  %idxprom79 = sext i32 %105 to i64
  %arrayidx80 = getelementptr inbounds i8, ptr %arrayidx78, i64 %idxprom79
  store i8 %conv72, ptr %arrayidx80, align 1
  br label %for.inc81

for.inc81:                                        ; preds = %cond.end70
  %106 = load i32, ptr %och, align 4
  %inc82 = add nsw i32 %106, 1
  store i32 %inc82, ptr %och, align 4
  br label %for.cond9, !llvm.loop !12

for.end83:                                        ; preds = %for.cond9
  br label %for.inc84

for.inc84:                                        ; preds = %for.end83
  %107 = load i32, ptr %ocol, align 4
  %inc85 = add nsw i32 %107, 1
  store i32 %inc85, ptr %ocol, align 4
  br label %for.cond6, !llvm.loop !13

for.end86:                                        ; preds = %for.cond6
  br label %for.inc87

for.inc87:                                        ; preds = %for.end86
  %108 = load i32, ptr %orow, align 4
  %inc88 = add nsw i32 %108, 1
  store i32 %inc88, ptr %orow, align 4
  br label %for.cond3, !llvm.loop !14

for.end89:                                        ; preds = %for.cond3
  br label %for.inc90

for.inc90:                                        ; preds = %for.end89
  %109 = load i32, ptr %b, align 4
  %inc91 = add nsw i32 %109, 1
  store i32 %inc91, ptr %b, align 4
  br label %for.cond, !llvm.loop !15

for.end92:                                        ; preds = %for.cond
  ret void
}

declare dso_local signext i32 @printf(ptr noundef, ...) #1

; Function Attrs: noreturn
declare dso_local void @exit(i32 noundef signext) #2

; Function Attrs: noinline nounwind optnone
define dso_local void @flatten_weights(i32 noundef signext %out_channels, i32 noundef signext %kernel_dim, i32 noundef signext %in_channels, i32 noundef signext %patch_size, ptr noundef %weights, ptr noundef %weights_mat) #0 {
entry:
  %out_channels.addr = alloca i32, align 4
  %kernel_dim.addr = alloca i32, align 4
  %in_channels.addr = alloca i32, align 4
  %patch_size.addr = alloca i32, align 4
  %weights.addr = alloca ptr, align 8
  %weights_mat.addr = alloca ptr, align 8
  %outc = alloca i32, align 4
  %krow = alloca i32, align 4
  %kcol = alloca i32, align 4
  %inc = alloca i32, align 4
  %wmatrow = alloca i32, align 4
  store i32 %out_channels, ptr %out_channels.addr, align 4
  store i32 %kernel_dim, ptr %kernel_dim.addr, align 4
  store i32 %in_channels, ptr %in_channels.addr, align 4
  store i32 %patch_size, ptr %patch_size.addr, align 4
  store ptr %weights, ptr %weights.addr, align 8
  store ptr %weights_mat, ptr %weights_mat.addr, align 8
  %0 = load i32, ptr %out_channels.addr, align 4
  %1 = zext i32 %0 to i64
  %2 = load i32, ptr %kernel_dim.addr, align 4
  %3 = zext i32 %2 to i64
  %4 = load i32, ptr %kernel_dim.addr, align 4
  %5 = zext i32 %4 to i64
  %6 = load i32, ptr %in_channels.addr, align 4
  %7 = zext i32 %6 to i64
  %8 = load i32, ptr %patch_size.addr, align 4
  %9 = zext i32 %8 to i64
  %10 = load i32, ptr %out_channels.addr, align 4
  %11 = zext i32 %10 to i64
  store i32 0, ptr %outc, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc30, %entry
  %12 = load i32, ptr %outc, align 4
  %13 = load i32, ptr %out_channels.addr, align 4
  %cmp = icmp slt i32 %12, %13
  br i1 %cmp, label %for.body, label %for.end32

for.body:                                         ; preds = %for.cond
  store i32 0, ptr %krow, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc27, %for.body
  %14 = load i32, ptr %krow, align 4
  %15 = load i32, ptr %kernel_dim.addr, align 4
  %cmp2 = icmp slt i32 %14, %15
  br i1 %cmp2, label %for.body3, label %for.end29

for.body3:                                        ; preds = %for.cond1
  store i32 0, ptr %kcol, align 4
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc24, %for.body3
  %16 = load i32, ptr %kcol, align 4
  %17 = load i32, ptr %kernel_dim.addr, align 4
  %cmp5 = icmp slt i32 %16, %17
  br i1 %cmp5, label %for.body6, label %for.end26

for.body6:                                        ; preds = %for.cond4
  store i32 0, ptr %inc, align 4
  br label %for.cond7

for.cond7:                                        ; preds = %for.inc, %for.body6
  %18 = load i32, ptr %inc, align 4
  %19 = load i32, ptr %in_channels.addr, align 4
  %cmp8 = icmp slt i32 %18, %19
  br i1 %cmp8, label %for.body9, label %for.end

for.body9:                                        ; preds = %for.cond7
  %20 = load i32, ptr %krow, align 4
  %21 = load i32, ptr %kernel_dim.addr, align 4
  %mul = mul nsw i32 %20, %21
  %22 = load i32, ptr %in_channels.addr, align 4
  %mul10 = mul nsw i32 %mul, %22
  %23 = load i32, ptr %kcol, align 4
  %24 = load i32, ptr %in_channels.addr, align 4
  %mul11 = mul nsw i32 %23, %24
  %add = add nsw i32 %mul10, %mul11
  %25 = load i32, ptr %inc, align 4
  %add12 = add nsw i32 %add, %25
  store i32 %add12, ptr %wmatrow, align 4
  %26 = load ptr, ptr %weights.addr, align 8
  %27 = load i32, ptr %outc, align 4
  %idxprom = sext i32 %27 to i64
  %28 = mul nuw i64 %3, %5
  %29 = mul nuw i64 %28, %7
  %30 = mul nsw i64 %idxprom, %29
  %arrayidx = getelementptr inbounds i8, ptr %26, i64 %30
  %31 = load i32, ptr %krow, align 4
  %idxprom13 = sext i32 %31 to i64
  %32 = mul nuw i64 %5, %7
  %33 = mul nsw i64 %idxprom13, %32
  %arrayidx14 = getelementptr inbounds i8, ptr %arrayidx, i64 %33
  %34 = load i32, ptr %kcol, align 4
  %idxprom15 = sext i32 %34 to i64
  %35 = mul nsw i64 %idxprom15, %7
  %arrayidx16 = getelementptr inbounds i8, ptr %arrayidx14, i64 %35
  %36 = load i32, ptr %inc, align 4
  %idxprom17 = sext i32 %36 to i64
  %arrayidx18 = getelementptr inbounds i8, ptr %arrayidx16, i64 %idxprom17
  %37 = load i8, ptr %arrayidx18, align 1
  %38 = load ptr, ptr %weights_mat.addr, align 8
  %39 = load i32, ptr %wmatrow, align 4
  %idxprom19 = sext i32 %39 to i64
  %40 = mul nsw i64 %idxprom19, %11
  %arrayidx20 = getelementptr inbounds i8, ptr %38, i64 %40
  %41 = load i32, ptr %outc, align 4
  %idxprom21 = sext i32 %41 to i64
  %arrayidx22 = getelementptr inbounds i8, ptr %arrayidx20, i64 %idxprom21
  store i8 %37, ptr %arrayidx22, align 1
  br label %for.inc

for.inc:                                          ; preds = %for.body9
  %42 = load i32, ptr %inc, align 4
  %inc23 = add nsw i32 %42, 1
  store i32 %inc23, ptr %inc, align 4
  br label %for.cond7, !llvm.loop !16

for.end:                                          ; preds = %for.cond7
  br label %for.inc24

for.inc24:                                        ; preds = %for.end
  %43 = load i32, ptr %kcol, align 4
  %inc25 = add nsw i32 %43, 1
  store i32 %inc25, ptr %kcol, align 4
  br label %for.cond4, !llvm.loop !17

for.end26:                                        ; preds = %for.cond4
  br label %for.inc27

for.inc27:                                        ; preds = %for.end26
  %44 = load i32, ptr %krow, align 4
  %inc28 = add nsw i32 %44, 1
  store i32 %inc28, ptr %krow, align 4
  br label %for.cond1, !llvm.loop !18

for.end29:                                        ; preds = %for.cond1
  br label %for.inc30

for.inc30:                                        ; preds = %for.end29
  %45 = load i32, ptr %outc, align 4
  %inc31 = add nsw i32 %45, 1
  store i32 %inc31, ptr %outc, align 4
  br label %for.cond, !llvm.loop !19

for.end32:                                        ; preds = %for.cond
  ret void
}

; Function Attrs: noinline nounwind optnone
define dso_local void @printf_vec(ptr noundef %buf, i32 noundef signext %len) #0 {
entry:
  %buf.addr = alloca ptr, align 8
  %len.addr = alloca i32, align 4
  %ptr = alloca ptr, align 8
  store ptr %buf, ptr %buf.addr, align 8
  store i32 %len, ptr %len.addr, align 4
  %0 = load ptr, ptr %buf.addr, align 8
  store ptr %0, ptr %ptr, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %1 = load ptr, ptr %ptr, align 8
  %2 = load ptr, ptr %buf.addr, align 8
  %3 = load i32, ptr %len.addr, align 4
  %idx.ext = sext i32 %3 to i64
  %add.ptr = getelementptr inbounds i8, ptr %2, i64 %idx.ext
  %cmp = icmp ult ptr %1, %add.ptr
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %4 = load ptr, ptr %ptr, align 8
  %5 = load i8, ptr %4, align 1
  %conv = sext i8 %5 to i32
  %call = call signext i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef signext %conv)
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %6 = load ptr, ptr %ptr, align 8
  %incdec.ptr = getelementptr inbounds i8, ptr %6, i32 1
  store ptr %incdec.ptr, ptr %ptr, align 8
  br label %for.cond, !llvm.loop !20

for.end:                                          ; preds = %for.cond
  %call1 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.2)
  ret void
}

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %input = alloca [2 x [17 x [17 x [18 x i8]]]], align 1
  %weights = alloca [19 x [3 x [3 x [18 x i8]]]], align 1
  %bias = alloca [19 x i32], align 4
  %output = alloca [2 x [9 x [9 x [19 x i8]]]], align 1
  %start_cpu = alloca i64, align 8
  %end_cpu = alloca i64, align 8
  %start_gemmini = alloca i64, align 8
  %end_gemmini = alloca i64, align 8
  store i32 0, ptr %retval, align 4
  %call = call signext i32 (ptr, ...) @printf(ptr noundef @.str.3)
  %arrayidx = getelementptr inbounds [2 x [17 x [17 x [18 x i8]]]], ptr %input, i64 0, i64 0
  %arrayidx1 = getelementptr inbounds [17 x [17 x [18 x i8]]], ptr %arrayidx, i64 0, i64 0
  %arrayidx2 = getelementptr inbounds [17 x [18 x i8]], ptr %arrayidx1, i64 0, i64 0
  %arrayidx3 = getelementptr inbounds [18 x i8], ptr %arrayidx2, i64 0, i64 0
  call void @init_random(ptr noundef %arrayidx3, i32 noundef signext 10404)
  %call4 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.4)
  %arrayidx5 = getelementptr inbounds [19 x [3 x [3 x [18 x i8]]]], ptr %weights, i64 0, i64 0
  %arrayidx6 = getelementptr inbounds [3 x [3 x [18 x i8]]], ptr %arrayidx5, i64 0, i64 0
  %arrayidx7 = getelementptr inbounds [3 x [18 x i8]], ptr %arrayidx6, i64 0, i64 0
  %arrayidx8 = getelementptr inbounds [18 x i8], ptr %arrayidx7, i64 0, i64 0
  call void @init_random(ptr noundef %arrayidx8, i32 noundef signext 3078)
  %call9 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.5)
  %arrayidx10 = getelementptr inbounds [19 x i32], ptr %bias, i64 0, i64 0
  call void @init_random_acc(ptr noundef %arrayidx10, i32 noundef signext 19)
  %call11 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.6)
  %call12 = call i64 @read_cycles()
  store i64 %call12, ptr %start_cpu, align 8
  %arraydecay = getelementptr inbounds [2 x [17 x [17 x [18 x i8]]]], ptr %input, i64 0, i64 0
  %arraydecay13 = getelementptr inbounds [19 x [3 x [3 x [18 x i8]]]], ptr %weights, i64 0, i64 0
  %arraydecay14 = getelementptr inbounds [19 x i32], ptr %bias, i64 0, i64 0
  %arraydecay15 = getelementptr inbounds [2 x [9 x [9 x [19 x i8]]]], ptr %output, i64 0, i64 0
  call void @conv(i32 noundef signext 2, i32 noundef signext 18, i32 noundef signext 17, i32 noundef signext 19, i32 noundef signext 3, i32 noundef signext 9, i32 noundef signext 2, i32 noundef signext 1, ptr noundef %arraydecay, ptr noundef %arraydecay13, ptr noundef %arraydecay14, ptr noundef %arraydecay15)
  %call16 = call i64 @read_cycles()
  store i64 %call16, ptr %end_cpu, align 8
  %0 = load i64, ptr %end_cpu, align 8
  %1 = load i64, ptr %start_cpu, align 8
  %sub = sub i64 %0, %1
  %call17 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.7, i64 noundef %sub)
  %call18 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.8)
  %arraydecay19 = getelementptr inbounds [19 x [3 x [3 x [18 x i8]]]], ptr %weights, i64 0, i64 0
  call void @flatten_weights(i32 noundef signext 19, i32 noundef signext 3, i32 noundef signext 18, i32 noundef signext 162, ptr noundef %arraydecay19, ptr noundef @main.weights_mat)
  %call20 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.9)
  %call21 = call i64 @read_cycles()
  store i64 %call21, ptr %start_gemmini, align 8
  %arraydecay22 = getelementptr inbounds [2 x [17 x [17 x [18 x i8]]]], ptr %input, i64 0, i64 0
  %arraydecay23 = getelementptr inbounds [19 x i32], ptr %bias, i64 0, i64 0
  call void @tiled_conv_auto(i32 noundef signext 2, i32 noundef signext 17, i32 noundef signext 18, i32 noundef signext 19, i32 noundef signext 9, i32 noundef signext 2, i32 noundef signext 1, i32 noundef signext 1, i32 noundef 1, i32 noundef 3, i1 noundef false, i1 noundef false, i1 noundef false, i1 noundef false, i1 noundef false, ptr noundef %arraydecay22, ptr noundef @main.weights_mat, ptr noundef %arraydecay23, ptr noundef @main.output_mat, i32 noundef 0, float noundef 1.000000e+00, i32 noundef 0, i32 noundef 0, i32 noundef 0, i32 noundef 1)
  %call24 = call i64 @read_cycles()
  store i64 %call24, ptr %end_gemmini, align 8
  %2 = load i64, ptr %end_gemmini, align 8
  %3 = load i64, ptr %start_gemmini, align 8
  %sub25 = sub i64 %2, %3
  %call26 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.10, i64 noundef %sub25)
  %call27 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.11)
  %arrayidx28 = getelementptr inbounds [2 x [9 x [9 x [19 x i8]]]], ptr %output, i64 0, i64 0
  %arrayidx29 = getelementptr inbounds [9 x [9 x [19 x i8]]], ptr %arrayidx28, i64 0, i64 0
  %arrayidx30 = getelementptr inbounds [9 x [19 x i8]], ptr %arrayidx29, i64 0, i64 0
  %arrayidx31 = getelementptr inbounds [19 x i8], ptr %arrayidx30, i64 0, i64 0
  call void @printf_vec(ptr noundef %arrayidx31, i32 noundef signext 3078)
  %call32 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.12)
  call void @printf_vec(ptr noundef @main.output_mat, i32 noundef signext 3078)
  ret i32 0
}

; Function Attrs: noinline nounwind optnone
define internal void @tiled_conv_auto(i32 noundef signext %batch_size, i32 noundef signext %in_dim, i32 noundef signext %in_channels, i32 noundef signext %out_channels, i32 noundef signext %out_dim, i32 noundef signext %stride, i32 noundef signext %input_dilation, i32 noundef signext %kernel_dilation, i32 noundef %padding, i32 noundef %kernel_dim, i1 noundef %wrot180, i1 noundef %trans_output_1203, i1 noundef %trans_input_3120, i1 noundef %trans_weight_1203, i1 noundef %trans_weight_0132, ptr noundef %input, ptr noundef %weights, ptr noundef %bias, ptr noundef %output, i32 noundef %act, float noundef %scale, i32 noundef %pool_size, i32 noundef %pool_stride, i32 noundef %pool_padding, i32 noundef %tiled_conv_type) #0 {
entry:
  %batch_size.addr = alloca i32, align 4
  %in_dim.addr = alloca i32, align 4
  %in_channels.addr = alloca i32, align 4
  %out_channels.addr = alloca i32, align 4
  %out_dim.addr = alloca i32, align 4
  %stride.addr = alloca i32, align 4
  %input_dilation.addr = alloca i32, align 4
  %kernel_dilation.addr = alloca i32, align 4
  %padding.addr = alloca i32, align 4
  %kernel_dim.addr = alloca i32, align 4
  %wrot180.addr = alloca i8, align 1
  %trans_output_1203.addr = alloca i8, align 1
  %trans_input_3120.addr = alloca i8, align 1
  %trans_weight_1203.addr = alloca i8, align 1
  %trans_weight_0132.addr = alloca i8, align 1
  %input.addr = alloca ptr, align 8
  %weights.addr = alloca ptr, align 8
  %bias.addr = alloca ptr, align 8
  %output.addr = alloca ptr, align 8
  %act.addr = alloca i32, align 4
  %scale.addr = alloca float, align 4
  %pool_size.addr = alloca i32, align 4
  %pool_stride.addr = alloca i32, align 4
  %pool_padding.addr = alloca i32, align 4
  %tiled_conv_type.addr = alloca i32, align 4
  %no_pool = alloca i8, align 1
  %pool_out_dim = alloca i32, align 4
  %downsample = alloca i8, align 1
  %args = alloca [7 x i32], align 4
  %max_args = alloca [7 x i32], align 4
  %orows_idx = alloca i32, align 4
  %ocols_idx = alloca i32, align 4
  %out_channels_idx = alloca i32, align 4
  %in_channels_idx = alloca i32, align 4
  %max_spad_rows = alloca i32, align 4
  %max_acc_rows = alloca i32, align 4
  %spad_rows = alloca i32, align 4
  %acc_rows = alloca i32, align 4
  %max_val = alloca i32, align 4
  %max_idx = alloca i32, align 4
  %i = alloca i64, align 8
  %not_increased = alloca i8, align 1
  %args_candidate = alloca [7 x i32], align 4
  %nothing_increased = alloca i8, align 1
  %i180 = alloca i64, align 8
  %args_candidate185 = alloca [7 x i32], align 4
  %batches = alloca i32, align 4
  %orows = alloca i32, align 4
  %ocols = alloca i32, align 4
  %ochs = alloca i32, align 4
  %krows = alloca i32, align 4
  %kcols = alloca i32, align 4
  %kchs = alloca i32, align 4
  store i32 %batch_size, ptr %batch_size.addr, align 4
  store i32 %in_dim, ptr %in_dim.addr, align 4
  store i32 %in_channels, ptr %in_channels.addr, align 4
  store i32 %out_channels, ptr %out_channels.addr, align 4
  store i32 %out_dim, ptr %out_dim.addr, align 4
  store i32 %stride, ptr %stride.addr, align 4
  store i32 %input_dilation, ptr %input_dilation.addr, align 4
  store i32 %kernel_dilation, ptr %kernel_dilation.addr, align 4
  store i32 %padding, ptr %padding.addr, align 4
  store i32 %kernel_dim, ptr %kernel_dim.addr, align 4
  %frombool = zext i1 %wrot180 to i8
  store i8 %frombool, ptr %wrot180.addr, align 1
  %frombool1 = zext i1 %trans_output_1203 to i8
  store i8 %frombool1, ptr %trans_output_1203.addr, align 1
  %frombool2 = zext i1 %trans_input_3120 to i8
  store i8 %frombool2, ptr %trans_input_3120.addr, align 1
  %frombool3 = zext i1 %trans_weight_1203 to i8
  store i8 %frombool3, ptr %trans_weight_1203.addr, align 1
  %frombool4 = zext i1 %trans_weight_0132 to i8
  store i8 %frombool4, ptr %trans_weight_0132.addr, align 1
  store ptr %input, ptr %input.addr, align 8
  store ptr %weights, ptr %weights.addr, align 8
  store ptr %bias, ptr %bias.addr, align 8
  store ptr %output, ptr %output.addr, align 8
  store i32 %act, ptr %act.addr, align 4
  store float %scale, ptr %scale.addr, align 4
  store i32 %pool_size, ptr %pool_size.addr, align 4
  store i32 %pool_stride, ptr %pool_stride.addr, align 4
  store i32 %pool_padding, ptr %pool_padding.addr, align 4
  store i32 %tiled_conv_type, ptr %tiled_conv_type.addr, align 4
  %call = call signext i32 (ptr, ...) @printf(ptr noundef @.str.13)
  %0 = load i32, ptr %batch_size.addr, align 4
  %1 = load i32, ptr %in_dim.addr, align 4
  %2 = load i32, ptr %in_channels.addr, align 4
  %3 = load i32, ptr %out_channels.addr, align 4
  %4 = load i32, ptr %out_dim.addr, align 4
  %5 = load i32, ptr %stride.addr, align 4
  %6 = load i32, ptr %input_dilation.addr, align 4
  %7 = load i32, ptr %kernel_dilation.addr, align 4
  %8 = load i32, ptr %padding.addr, align 4
  %9 = load i32, ptr %kernel_dim.addr, align 4
  %call5 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.14, i32 noundef signext %0, i32 noundef signext %1, i32 noundef signext %2, i32 noundef signext %3, i32 noundef signext %4, i32 noundef signext %5, i32 noundef signext %6, i32 noundef %7, i32 noundef %8, i32 noundef %9)
  %10 = load i32, ptr %pool_stride.addr, align 4
  %cmp = icmp eq i32 %10, 0
  %frombool6 = zext i1 %cmp to i8
  store i8 %frombool6, ptr %no_pool, align 1
  %11 = load i8, ptr %no_pool, align 1
  %tobool = trunc i8 %11 to i1
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 1, ptr %pool_size.addr, align 4
  store i32 1, ptr %pool_stride.addr, align 4
  store i32 0, ptr %pool_padding.addr, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %12 = load i32, ptr %out_dim.addr, align 4
  %13 = load i32, ptr %pool_padding.addr, align 4
  %mul = mul nsw i32 2, %13
  %add = add nsw i32 %12, %mul
  %14 = load i32, ptr %pool_size.addr, align 4
  %sub = sub nsw i32 %add, %14
  %15 = load i32, ptr %pool_stride.addr, align 4
  %div = sdiv i32 %sub, %15
  %add7 = add nsw i32 %div, 1
  store i32 %add7, ptr %pool_out_dim, align 4
  %16 = load i32, ptr %stride.addr, align 4
  %cmp8 = icmp eq i32 %16, 2
  br i1 %cmp8, label %land.lhs.true, label %land.end

land.lhs.true:                                    ; preds = %if.end
  %17 = load i32, ptr %kernel_dim.addr, align 4
  %cmp9 = icmp eq i32 %17, 1
  br i1 %cmp9, label %land.lhs.true10, label %land.end

land.lhs.true10:                                  ; preds = %land.lhs.true
  %18 = load i32, ptr %padding.addr, align 4
  %cmp11 = icmp eq i32 %18, 0
  br i1 %cmp11, label %land.lhs.true12, label %land.end

land.lhs.true12:                                  ; preds = %land.lhs.true10
  %19 = load i8, ptr %no_pool, align 1
  %tobool13 = trunc i8 %19 to i1
  br i1 %tobool13, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %land.lhs.true12
  %20 = load i32, ptr %in_dim.addr, align 4
  %rem = srem i32 %20, 2
  %cmp14 = icmp eq i32 %rem, 0
  br label %land.end

land.end:                                         ; preds = %land.rhs, %land.lhs.true12, %land.lhs.true10, %land.lhs.true, %if.end
  %21 = phi i1 [ false, %land.lhs.true12 ], [ false, %land.lhs.true10 ], [ false, %land.lhs.true ], [ false, %if.end ], [ %cmp14, %land.rhs ]
  %frombool15 = zext i1 %21 to i8
  store i8 %frombool15, ptr %downsample, align 1
  %arrayinit.begin = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 0
  %22 = load i32, ptr %batch_size.addr, align 4
  store i32 %22, ptr %arrayinit.begin, align 4
  %arrayinit.element = getelementptr inbounds i32, ptr %arrayinit.begin, i64 1
  %23 = load i32, ptr %pool_out_dim, align 4
  store i32 %23, ptr %arrayinit.element, align 4
  %arrayinit.element16 = getelementptr inbounds i32, ptr %arrayinit.element, i64 1
  %24 = load i32, ptr %pool_out_dim, align 4
  store i32 %24, ptr %arrayinit.element16, align 4
  %arrayinit.element17 = getelementptr inbounds i32, ptr %arrayinit.element16, i64 1
  %25 = load i32, ptr %out_channels.addr, align 4
  store i32 %25, ptr %arrayinit.element17, align 4
  %arrayinit.element18 = getelementptr inbounds i32, ptr %arrayinit.element17, i64 1
  %26 = load i32, ptr %kernel_dim.addr, align 4
  store i32 %26, ptr %arrayinit.element18, align 4
  %arrayinit.element19 = getelementptr inbounds i32, ptr %arrayinit.element18, i64 1
  %27 = load i32, ptr %kernel_dim.addr, align 4
  store i32 %27, ptr %arrayinit.element19, align 4
  %arrayinit.element20 = getelementptr inbounds i32, ptr %arrayinit.element19, i64 1
  %28 = load i32, ptr %in_channels.addr, align 4
  store i32 %28, ptr %arrayinit.element20, align 4
  %arrayinit.begin21 = getelementptr inbounds [7 x i32], ptr %max_args, i64 0, i64 0
  %29 = load i32, ptr %batch_size.addr, align 4
  store i32 %29, ptr %arrayinit.begin21, align 4
  %arrayinit.element22 = getelementptr inbounds i32, ptr %arrayinit.begin21, i64 1
  %30 = load i32, ptr %pool_out_dim, align 4
  store i32 %30, ptr %arrayinit.element22, align 4
  %arrayinit.element23 = getelementptr inbounds i32, ptr %arrayinit.element22, i64 1
  %31 = load i32, ptr %pool_out_dim, align 4
  store i32 %31, ptr %arrayinit.element23, align 4
  %arrayinit.element24 = getelementptr inbounds i32, ptr %arrayinit.element23, i64 1
  %32 = load i32, ptr %out_channels.addr, align 4
  store i32 %32, ptr %arrayinit.element24, align 4
  %arrayinit.element25 = getelementptr inbounds i32, ptr %arrayinit.element24, i64 1
  %33 = load i32, ptr %kernel_dim.addr, align 4
  store i32 %33, ptr %arrayinit.element25, align 4
  %arrayinit.element26 = getelementptr inbounds i32, ptr %arrayinit.element25, i64 1
  %34 = load i32, ptr %kernel_dim.addr, align 4
  store i32 %34, ptr %arrayinit.element26, align 4
  %arrayinit.element27 = getelementptr inbounds i32, ptr %arrayinit.element26, i64 1
  %35 = load i32, ptr %in_channels.addr, align 4
  store i32 %35, ptr %arrayinit.element27, align 4
  store i32 1, ptr %orows_idx, align 4
  store i32 2, ptr %ocols_idx, align 4
  store i32 3, ptr %out_channels_idx, align 4
  store i32 6, ptr %in_channels_idx, align 4
  store i32 8192, ptr %max_spad_rows, align 4
  store i32 512, ptr %max_acc_rows, align 4
  %36 = load i32, ptr %stride.addr, align 4
  %37 = load i32, ptr %input_dilation.addr, align 4
  %38 = load i32, ptr %kernel_dilation.addr, align 4
  %39 = load i8, ptr %downsample, align 1
  %tobool28 = trunc i8 %39 to i1
  %40 = load i8, ptr %trans_weight_0132.addr, align 1
  %tobool29 = trunc i8 %40 to i1
  %41 = load i8, ptr %trans_input_3120.addr, align 1
  %tobool30 = trunc i8 %41 to i1
  %arrayidx = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 0
  %42 = load i32, ptr %arrayidx, align 4
  %arrayidx31 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 1
  %43 = load i32, ptr %arrayidx31, align 4
  %arrayidx32 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 2
  %44 = load i32, ptr %arrayidx32, align 4
  %arrayidx33 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 3
  %45 = load i32, ptr %arrayidx33, align 4
  %arrayidx34 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 4
  %46 = load i32, ptr %arrayidx34, align 4
  %arrayidx35 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 5
  %47 = load i32, ptr %arrayidx35, align 4
  %arrayidx36 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 6
  %48 = load i32, ptr %arrayidx36, align 4
  %49 = load i32, ptr %pool_size.addr, align 4
  %50 = load i32, ptr %pool_stride.addr, align 4
  %call37 = call signext i32 @tiled_conv_total_spad_rows(i1 noundef zeroext false, i32 noundef signext %36, i32 noundef signext %37, i32 noundef signext %38, i1 noundef zeroext %tobool28, i1 noundef zeroext %tobool29, i1 noundef zeroext %tobool30, i32 noundef signext %42, i32 noundef %43, i32 noundef %44, i32 noundef %45, i32 noundef %46, i32 noundef %47, i32 noundef %48, i32 noundef %49, i32 noundef %50)
  store i32 %call37, ptr %spad_rows, align 4
  %51 = load i32, ptr %stride.addr, align 4
  %52 = load i32, ptr %input_dilation.addr, align 4
  %53 = load i32, ptr %kernel_dilation.addr, align 4
  %54 = load i8, ptr %downsample, align 1
  %tobool38 = trunc i8 %54 to i1
  %55 = load i8, ptr %trans_weight_0132.addr, align 1
  %tobool39 = trunc i8 %55 to i1
  %56 = load i8, ptr %trans_input_3120.addr, align 1
  %tobool40 = trunc i8 %56 to i1
  %arrayidx41 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 0
  %57 = load i32, ptr %arrayidx41, align 4
  %arrayidx42 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 1
  %58 = load i32, ptr %arrayidx42, align 4
  %arrayidx43 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 2
  %59 = load i32, ptr %arrayidx43, align 4
  %arrayidx44 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 3
  %60 = load i32, ptr %arrayidx44, align 4
  %arrayidx45 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 4
  %61 = load i32, ptr %arrayidx45, align 4
  %arrayidx46 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 5
  %62 = load i32, ptr %arrayidx46, align 4
  %arrayidx47 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 6
  %63 = load i32, ptr %arrayidx47, align 4
  %64 = load i32, ptr %pool_size.addr, align 4
  %65 = load i32, ptr %pool_stride.addr, align 4
  %call48 = call signext i32 @tiled_conv_total_spad_rows(i1 noundef zeroext true, i32 noundef signext %51, i32 noundef signext %52, i32 noundef signext %53, i1 noundef zeroext %tobool38, i1 noundef zeroext %tobool39, i1 noundef zeroext %tobool40, i32 noundef signext %57, i32 noundef %58, i32 noundef %59, i32 noundef %60, i32 noundef %61, i32 noundef %62, i32 noundef %63, i32 noundef %64, i32 noundef %65)
  store i32 %call48, ptr %acc_rows, align 4
  br label %while.cond

while.cond:                                       ; preds = %if.end96, %land.end
  %66 = load i32, ptr %spad_rows, align 4
  %cmp49 = icmp sgt i32 %66, 8192
  br i1 %cmp49, label %lor.end, label %lor.rhs

lor.rhs:                                          ; preds = %while.cond
  %67 = load i32, ptr %acc_rows, align 4
  %cmp50 = icmp sgt i32 %67, 512
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %while.cond
  %68 = phi i1 [ true, %while.cond ], [ %cmp50, %lor.rhs ]
  br i1 %68, label %while.body, label %while.end

while.body:                                       ; preds = %lor.end
  store i32 -1, ptr %max_val, align 4
  store i32 -1, ptr %max_idx, align 4
  store i64 0, ptr %i, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %while.body
  %69 = load i64, ptr %i, align 8
  %cmp51 = icmp ult i64 %69, 7
  br i1 %cmp51, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %70 = load i64, ptr %i, align 8
  %cmp52 = icmp eq i64 %70, 2
  br i1 %cmp52, label %land.lhs.true53, label %land.lhs.true59

land.lhs.true53:                                  ; preds = %for.body
  %71 = load i64, ptr %i, align 8
  %arrayidx54 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 %71
  %72 = load i32, ptr %arrayidx54, align 4
  %cmp55 = icmp sle i32 %72, 16
  br i1 %cmp55, label %land.lhs.true56, label %land.lhs.true59

land.lhs.true56:                                  ; preds = %land.lhs.true53
  %arrayidx57 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 1
  %73 = load i32, ptr %arrayidx57, align 4
  %cmp58 = icmp sgt i32 %73, 1
  br i1 %cmp58, label %if.end64, label %land.lhs.true59

land.lhs.true59:                                  ; preds = %land.lhs.true56, %land.lhs.true53, %for.body
  %74 = load i64, ptr %i, align 8
  %arrayidx60 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 %74
  %75 = load i32, ptr %arrayidx60, align 4
  %76 = load i32, ptr %max_val, align 4
  %cmp61 = icmp sgt i32 %75, %76
  br i1 %cmp61, label %if.then62, label %if.end64

if.then62:                                        ; preds = %land.lhs.true59
  %77 = load i64, ptr %i, align 8
  %arrayidx63 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 %77
  %78 = load i32, ptr %arrayidx63, align 4
  store i32 %78, ptr %max_val, align 4
  %79 = load i64, ptr %i, align 8
  %conv = trunc i64 %79 to i32
  store i32 %conv, ptr %max_idx, align 4
  br label %if.end64

if.end64:                                         ; preds = %if.then62, %land.lhs.true59, %land.lhs.true56
  br label %for.inc

for.inc:                                          ; preds = %if.end64
  %80 = load i64, ptr %i, align 8
  %inc = add i64 %80, 1
  store i64 %inc, ptr %i, align 8
  br label %for.cond, !llvm.loop !21

for.end:                                          ; preds = %for.cond
  %81 = load i32, ptr %max_idx, align 4
  %cmp65 = icmp eq i32 %81, 3
  br i1 %cmp65, label %if.then69, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %for.end
  %82 = load i32, ptr %max_idx, align 4
  %cmp67 = icmp eq i32 %82, 6
  br i1 %cmp67, label %if.then69, label %if.else93

if.then69:                                        ; preds = %lor.lhs.false, %for.end
  %83 = load i32, ptr %max_idx, align 4
  %idxprom = sext i32 %83 to i64
  %arrayidx70 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 %idxprom
  %84 = load i32, ptr %arrayidx70, align 4
  %rem71 = srem i32 %84, 16
  %cmp72 = icmp ne i32 %rem71, 0
  br i1 %cmp72, label %if.then74, label %if.else

if.then74:                                        ; preds = %if.then69
  %85 = load i32, ptr %max_idx, align 4
  %idxprom75 = sext i32 %85 to i64
  %arrayidx76 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 %idxprom75
  %86 = load i32, ptr %arrayidx76, align 4
  %div77 = sdiv i32 %86, 16
  %mul78 = mul nsw i32 %div77, 16
  %87 = load i32, ptr %max_idx, align 4
  %idxprom79 = sext i32 %87 to i64
  %arrayidx80 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 %idxprom79
  store i32 %mul78, ptr %arrayidx80, align 4
  br label %if.end84

if.else:                                          ; preds = %if.then69
  %88 = load i32, ptr %max_idx, align 4
  %idxprom81 = sext i32 %88 to i64
  %arrayidx82 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 %idxprom81
  %89 = load i32, ptr %arrayidx82, align 4
  %sub83 = sub nsw i32 %89, 16
  store i32 %sub83, ptr %arrayidx82, align 4
  br label %if.end84

if.end84:                                         ; preds = %if.else, %if.then74
  %90 = load i32, ptr %max_idx, align 4
  %idxprom85 = sext i32 %90 to i64
  %arrayidx86 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 %idxprom85
  %91 = load i32, ptr %arrayidx86, align 4
  %cmp87 = icmp eq i32 %91, 0
  br i1 %cmp87, label %cond.true, label %cond.false

cond.true:                                        ; preds = %if.end84
  br label %cond.end

cond.false:                                       ; preds = %if.end84
  %92 = load i32, ptr %max_idx, align 4
  %idxprom89 = sext i32 %92 to i64
  %arrayidx90 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 %idxprom89
  %93 = load i32, ptr %arrayidx90, align 4
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ 1, %cond.true ], [ %93, %cond.false ]
  %94 = load i32, ptr %max_idx, align 4
  %idxprom91 = sext i32 %94 to i64
  %arrayidx92 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 %idxprom91
  store i32 %cond, ptr %arrayidx92, align 4
  br label %if.end96

if.else93:                                        ; preds = %lor.lhs.false
  %95 = load i32, ptr %max_idx, align 4
  %idxprom94 = sext i32 %95 to i64
  %arrayidx95 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 %idxprom94
  %96 = load i32, ptr %arrayidx95, align 4
  %dec = add nsw i32 %96, -1
  store i32 %dec, ptr %arrayidx95, align 4
  br label %if.end96

if.end96:                                         ; preds = %if.else93, %cond.end
  %97 = load i32, ptr %stride.addr, align 4
  %98 = load i32, ptr %input_dilation.addr, align 4
  %99 = load i32, ptr %kernel_dilation.addr, align 4
  %100 = load i8, ptr %downsample, align 1
  %tobool97 = trunc i8 %100 to i1
  %101 = load i8, ptr %trans_weight_0132.addr, align 1
  %tobool98 = trunc i8 %101 to i1
  %102 = load i8, ptr %trans_input_3120.addr, align 1
  %tobool99 = trunc i8 %102 to i1
  %arrayidx100 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 0
  %103 = load i32, ptr %arrayidx100, align 4
  %arrayidx101 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 1
  %104 = load i32, ptr %arrayidx101, align 4
  %arrayidx102 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 2
  %105 = load i32, ptr %arrayidx102, align 4
  %arrayidx103 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 3
  %106 = load i32, ptr %arrayidx103, align 4
  %arrayidx104 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 4
  %107 = load i32, ptr %arrayidx104, align 4
  %arrayidx105 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 5
  %108 = load i32, ptr %arrayidx105, align 4
  %arrayidx106 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 6
  %109 = load i32, ptr %arrayidx106, align 4
  %110 = load i32, ptr %pool_size.addr, align 4
  %111 = load i32, ptr %pool_stride.addr, align 4
  %call107 = call signext i32 @tiled_conv_total_spad_rows(i1 noundef zeroext false, i32 noundef signext %97, i32 noundef signext %98, i32 noundef signext %99, i1 noundef zeroext %tobool97, i1 noundef zeroext %tobool98, i1 noundef zeroext %tobool99, i32 noundef signext %103, i32 noundef %104, i32 noundef %105, i32 noundef %106, i32 noundef %107, i32 noundef %108, i32 noundef %109, i32 noundef %110, i32 noundef %111)
  store i32 %call107, ptr %spad_rows, align 4
  %112 = load i32, ptr %stride.addr, align 4
  %113 = load i32, ptr %input_dilation.addr, align 4
  %114 = load i32, ptr %kernel_dilation.addr, align 4
  %115 = load i8, ptr %downsample, align 1
  %tobool108 = trunc i8 %115 to i1
  %116 = load i8, ptr %trans_weight_0132.addr, align 1
  %tobool109 = trunc i8 %116 to i1
  %117 = load i8, ptr %trans_input_3120.addr, align 1
  %tobool110 = trunc i8 %117 to i1
  %arrayidx111 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 0
  %118 = load i32, ptr %arrayidx111, align 4
  %arrayidx112 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 1
  %119 = load i32, ptr %arrayidx112, align 4
  %arrayidx113 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 2
  %120 = load i32, ptr %arrayidx113, align 4
  %arrayidx114 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 3
  %121 = load i32, ptr %arrayidx114, align 4
  %arrayidx115 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 4
  %122 = load i32, ptr %arrayidx115, align 4
  %arrayidx116 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 5
  %123 = load i32, ptr %arrayidx116, align 4
  %arrayidx117 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 6
  %124 = load i32, ptr %arrayidx117, align 4
  %125 = load i32, ptr %pool_size.addr, align 4
  %126 = load i32, ptr %pool_stride.addr, align 4
  %call118 = call signext i32 @tiled_conv_total_spad_rows(i1 noundef zeroext true, i32 noundef signext %112, i32 noundef signext %113, i32 noundef signext %114, i1 noundef zeroext %tobool108, i1 noundef zeroext %tobool109, i1 noundef zeroext %tobool110, i32 noundef signext %118, i32 noundef %119, i32 noundef %120, i32 noundef %121, i32 noundef %122, i32 noundef %123, i32 noundef %124, i32 noundef %125, i32 noundef %126)
  store i32 %call118, ptr %acc_rows, align 4
  br label %while.cond, !llvm.loop !22

while.end:                                        ; preds = %lor.end
  store i8 0, ptr %not_increased, align 1
  br label %while.cond119

while.cond119:                                    ; preds = %if.end174, %if.then142, %while.end
  %127 = load i8, ptr %not_increased, align 1
  %tobool120 = trunc i8 %127 to i1
  %lnot = xor i1 %tobool120, true
  br i1 %lnot, label %while.body121, label %while.end175

while.body121:                                    ; preds = %while.cond119
  store i8 1, ptr %not_increased, align 1
  %arrayinit.begin122 = getelementptr inbounds [7 x i32], ptr %args_candidate, i64 0, i64 0
  %arrayidx123 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 0
  %128 = load i32, ptr %arrayidx123, align 4
  store i32 %128, ptr %arrayinit.begin122, align 4
  %arrayinit.element124 = getelementptr inbounds i32, ptr %arrayinit.begin122, i64 1
  %arrayidx125 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 1
  %129 = load i32, ptr %arrayidx125, align 4
  store i32 %129, ptr %arrayinit.element124, align 4
  %arrayinit.element126 = getelementptr inbounds i32, ptr %arrayinit.element124, i64 1
  %arrayidx127 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 2
  %130 = load i32, ptr %arrayidx127, align 4
  store i32 %130, ptr %arrayinit.element126, align 4
  %arrayinit.element128 = getelementptr inbounds i32, ptr %arrayinit.element126, i64 1
  %arrayidx129 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 3
  %131 = load i32, ptr %arrayidx129, align 4
  store i32 %131, ptr %arrayinit.element128, align 4
  %arrayinit.element130 = getelementptr inbounds i32, ptr %arrayinit.element128, i64 1
  %arrayidx131 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 4
  %132 = load i32, ptr %arrayidx131, align 4
  store i32 %132, ptr %arrayinit.element130, align 4
  %arrayinit.element132 = getelementptr inbounds i32, ptr %arrayinit.element130, i64 1
  %arrayidx133 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 5
  %133 = load i32, ptr %arrayidx133, align 4
  store i32 %133, ptr %arrayinit.element132, align 4
  %arrayinit.element134 = getelementptr inbounds i32, ptr %arrayinit.element132, i64 1
  %arrayidx135 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 6
  %134 = load i32, ptr %arrayidx135, align 4
  store i32 %134, ptr %arrayinit.element134, align 4
  %arrayidx136 = getelementptr inbounds [7 x i32], ptr %args_candidate, i64 0, i64 2
  %135 = load i32, ptr %arrayidx136, align 4
  %inc137 = add nsw i32 %135, 1
  store i32 %inc137, ptr %arrayidx136, align 4
  %arrayidx138 = getelementptr inbounds [7 x i32], ptr %args_candidate, i64 0, i64 2
  %136 = load i32, ptr %arrayidx138, align 4
  %arrayidx139 = getelementptr inbounds [7 x i32], ptr %max_args, i64 0, i64 2
  %137 = load i32, ptr %arrayidx139, align 4
  %cmp140 = icmp sgt i32 %136, %137
  br i1 %cmp140, label %if.then142, label %if.end143

if.then142:                                       ; preds = %while.body121
  br label %while.cond119, !llvm.loop !23

if.end143:                                        ; preds = %while.body121
  %138 = load i32, ptr %stride.addr, align 4
  %139 = load i32, ptr %input_dilation.addr, align 4
  %140 = load i32, ptr %kernel_dilation.addr, align 4
  %141 = load i8, ptr %downsample, align 1
  %tobool144 = trunc i8 %141 to i1
  %142 = load i8, ptr %trans_weight_0132.addr, align 1
  %tobool145 = trunc i8 %142 to i1
  %143 = load i8, ptr %trans_input_3120.addr, align 1
  %tobool146 = trunc i8 %143 to i1
  %arrayidx147 = getelementptr inbounds [7 x i32], ptr %args_candidate, i64 0, i64 0
  %144 = load i32, ptr %arrayidx147, align 4
  %arrayidx148 = getelementptr inbounds [7 x i32], ptr %args_candidate, i64 0, i64 1
  %145 = load i32, ptr %arrayidx148, align 4
  %arrayidx149 = getelementptr inbounds [7 x i32], ptr %args_candidate, i64 0, i64 2
  %146 = load i32, ptr %arrayidx149, align 4
  %arrayidx150 = getelementptr inbounds [7 x i32], ptr %args_candidate, i64 0, i64 3
  %147 = load i32, ptr %arrayidx150, align 4
  %arrayidx151 = getelementptr inbounds [7 x i32], ptr %args_candidate, i64 0, i64 4
  %148 = load i32, ptr %arrayidx151, align 4
  %arrayidx152 = getelementptr inbounds [7 x i32], ptr %args_candidate, i64 0, i64 5
  %149 = load i32, ptr %arrayidx152, align 4
  %arrayidx153 = getelementptr inbounds [7 x i32], ptr %args_candidate, i64 0, i64 6
  %150 = load i32, ptr %arrayidx153, align 4
  %151 = load i32, ptr %pool_size.addr, align 4
  %152 = load i32, ptr %pool_stride.addr, align 4
  %call154 = call signext i32 @tiled_conv_total_spad_rows(i1 noundef zeroext false, i32 noundef signext %138, i32 noundef signext %139, i32 noundef signext %140, i1 noundef zeroext %tobool144, i1 noundef zeroext %tobool145, i1 noundef zeroext %tobool146, i32 noundef signext %144, i32 noundef %145, i32 noundef %146, i32 noundef %147, i32 noundef %148, i32 noundef %149, i32 noundef %150, i32 noundef %151, i32 noundef %152)
  store i32 %call154, ptr %spad_rows, align 4
  %153 = load i32, ptr %stride.addr, align 4
  %154 = load i32, ptr %input_dilation.addr, align 4
  %155 = load i32, ptr %kernel_dilation.addr, align 4
  %156 = load i8, ptr %downsample, align 1
  %tobool155 = trunc i8 %156 to i1
  %157 = load i8, ptr %trans_weight_0132.addr, align 1
  %tobool156 = trunc i8 %157 to i1
  %158 = load i8, ptr %trans_input_3120.addr, align 1
  %tobool157 = trunc i8 %158 to i1
  %arrayidx158 = getelementptr inbounds [7 x i32], ptr %args_candidate, i64 0, i64 0
  %159 = load i32, ptr %arrayidx158, align 4
  %arrayidx159 = getelementptr inbounds [7 x i32], ptr %args_candidate, i64 0, i64 1
  %160 = load i32, ptr %arrayidx159, align 4
  %arrayidx160 = getelementptr inbounds [7 x i32], ptr %args_candidate, i64 0, i64 2
  %161 = load i32, ptr %arrayidx160, align 4
  %arrayidx161 = getelementptr inbounds [7 x i32], ptr %args_candidate, i64 0, i64 3
  %162 = load i32, ptr %arrayidx161, align 4
  %arrayidx162 = getelementptr inbounds [7 x i32], ptr %args_candidate, i64 0, i64 4
  %163 = load i32, ptr %arrayidx162, align 4
  %arrayidx163 = getelementptr inbounds [7 x i32], ptr %args_candidate, i64 0, i64 5
  %164 = load i32, ptr %arrayidx163, align 4
  %arrayidx164 = getelementptr inbounds [7 x i32], ptr %args_candidate, i64 0, i64 6
  %165 = load i32, ptr %arrayidx164, align 4
  %166 = load i32, ptr %pool_size.addr, align 4
  %167 = load i32, ptr %pool_stride.addr, align 4
  %call165 = call signext i32 @tiled_conv_total_spad_rows(i1 noundef zeroext true, i32 noundef signext %153, i32 noundef signext %154, i32 noundef signext %155, i1 noundef zeroext %tobool155, i1 noundef zeroext %tobool156, i1 noundef zeroext %tobool157, i32 noundef signext %159, i32 noundef %160, i32 noundef %161, i32 noundef %162, i32 noundef %163, i32 noundef %164, i32 noundef %165, i32 noundef %166, i32 noundef %167)
  store i32 %call165, ptr %acc_rows, align 4
  %168 = load i32, ptr %spad_rows, align 4
  %cmp166 = icmp sle i32 %168, 8192
  br i1 %cmp166, label %land.lhs.true168, label %if.end174

land.lhs.true168:                                 ; preds = %if.end143
  %169 = load i32, ptr %acc_rows, align 4
  %cmp169 = icmp sle i32 %169, 512
  br i1 %cmp169, label %if.then171, label %if.end174

if.then171:                                       ; preds = %land.lhs.true168
  %arrayidx172 = getelementptr inbounds [7 x i32], ptr %args_candidate, i64 0, i64 2
  %170 = load i32, ptr %arrayidx172, align 4
  %arrayidx173 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 2
  store i32 %170, ptr %arrayidx173, align 4
  store i8 0, ptr %not_increased, align 1
  br label %if.end174

if.end174:                                        ; preds = %if.then171, %land.lhs.true168, %if.end143
  br label %while.cond119, !llvm.loop !23

while.end175:                                     ; preds = %while.cond119
  store i8 0, ptr %nothing_increased, align 1
  br label %while.cond176

while.cond176:                                    ; preds = %for.end241, %while.end175
  %171 = load i8, ptr %nothing_increased, align 1
  %tobool177 = trunc i8 %171 to i1
  %lnot178 = xor i1 %tobool177, true
  br i1 %lnot178, label %while.body179, label %while.end242

while.body179:                                    ; preds = %while.cond176
  store i8 1, ptr %nothing_increased, align 1
  store i64 0, ptr %i180, align 8
  br label %for.cond181

for.cond181:                                      ; preds = %for.inc239, %while.body179
  %172 = load i64, ptr %i180, align 8
  %cmp182 = icmp ult i64 %172, 7
  br i1 %cmp182, label %for.body184, label %for.end241

for.body184:                                      ; preds = %for.cond181
  %arrayinit.begin186 = getelementptr inbounds [7 x i32], ptr %args_candidate185, i64 0, i64 0
  %arrayidx187 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 0
  %173 = load i32, ptr %arrayidx187, align 4
  store i32 %173, ptr %arrayinit.begin186, align 4
  %arrayinit.element188 = getelementptr inbounds i32, ptr %arrayinit.begin186, i64 1
  %arrayidx189 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 1
  %174 = load i32, ptr %arrayidx189, align 4
  store i32 %174, ptr %arrayinit.element188, align 4
  %arrayinit.element190 = getelementptr inbounds i32, ptr %arrayinit.element188, i64 1
  %arrayidx191 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 2
  %175 = load i32, ptr %arrayidx191, align 4
  store i32 %175, ptr %arrayinit.element190, align 4
  %arrayinit.element192 = getelementptr inbounds i32, ptr %arrayinit.element190, i64 1
  %arrayidx193 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 3
  %176 = load i32, ptr %arrayidx193, align 4
  store i32 %176, ptr %arrayinit.element192, align 4
  %arrayinit.element194 = getelementptr inbounds i32, ptr %arrayinit.element192, i64 1
  %arrayidx195 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 4
  %177 = load i32, ptr %arrayidx195, align 4
  store i32 %177, ptr %arrayinit.element194, align 4
  %arrayinit.element196 = getelementptr inbounds i32, ptr %arrayinit.element194, i64 1
  %arrayidx197 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 5
  %178 = load i32, ptr %arrayidx197, align 4
  store i32 %178, ptr %arrayinit.element196, align 4
  %arrayinit.element198 = getelementptr inbounds i32, ptr %arrayinit.element196, i64 1
  %arrayidx199 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 6
  %179 = load i32, ptr %arrayidx199, align 4
  store i32 %179, ptr %arrayinit.element198, align 4
  %180 = load i64, ptr %i180, align 8
  %arrayidx200 = getelementptr inbounds [7 x i32], ptr %args_candidate185, i64 0, i64 %180
  %181 = load i32, ptr %arrayidx200, align 4
  %inc201 = add nsw i32 %181, 1
  store i32 %inc201, ptr %arrayidx200, align 4
  %182 = load i64, ptr %i180, align 8
  %arrayidx202 = getelementptr inbounds [7 x i32], ptr %args_candidate185, i64 0, i64 %182
  %183 = load i32, ptr %arrayidx202, align 4
  %184 = load i64, ptr %i180, align 8
  %arrayidx203 = getelementptr inbounds [7 x i32], ptr %max_args, i64 0, i64 %184
  %185 = load i32, ptr %arrayidx203, align 4
  %cmp204 = icmp sgt i32 %183, %185
  br i1 %cmp204, label %if.then206, label %if.end207

if.then206:                                       ; preds = %for.body184
  br label %for.inc239

if.end207:                                        ; preds = %for.body184
  %186 = load i32, ptr %stride.addr, align 4
  %187 = load i32, ptr %input_dilation.addr, align 4
  %188 = load i32, ptr %kernel_dilation.addr, align 4
  %189 = load i8, ptr %downsample, align 1
  %tobool208 = trunc i8 %189 to i1
  %190 = load i8, ptr %trans_weight_0132.addr, align 1
  %tobool209 = trunc i8 %190 to i1
  %191 = load i8, ptr %trans_input_3120.addr, align 1
  %tobool210 = trunc i8 %191 to i1
  %arrayidx211 = getelementptr inbounds [7 x i32], ptr %args_candidate185, i64 0, i64 0
  %192 = load i32, ptr %arrayidx211, align 4
  %arrayidx212 = getelementptr inbounds [7 x i32], ptr %args_candidate185, i64 0, i64 1
  %193 = load i32, ptr %arrayidx212, align 4
  %arrayidx213 = getelementptr inbounds [7 x i32], ptr %args_candidate185, i64 0, i64 2
  %194 = load i32, ptr %arrayidx213, align 4
  %arrayidx214 = getelementptr inbounds [7 x i32], ptr %args_candidate185, i64 0, i64 3
  %195 = load i32, ptr %arrayidx214, align 4
  %arrayidx215 = getelementptr inbounds [7 x i32], ptr %args_candidate185, i64 0, i64 4
  %196 = load i32, ptr %arrayidx215, align 4
  %arrayidx216 = getelementptr inbounds [7 x i32], ptr %args_candidate185, i64 0, i64 5
  %197 = load i32, ptr %arrayidx216, align 4
  %arrayidx217 = getelementptr inbounds [7 x i32], ptr %args_candidate185, i64 0, i64 6
  %198 = load i32, ptr %arrayidx217, align 4
  %199 = load i32, ptr %pool_size.addr, align 4
  %200 = load i32, ptr %pool_stride.addr, align 4
  %call218 = call signext i32 @tiled_conv_total_spad_rows(i1 noundef zeroext false, i32 noundef signext %186, i32 noundef signext %187, i32 noundef signext %188, i1 noundef zeroext %tobool208, i1 noundef zeroext %tobool209, i1 noundef zeroext %tobool210, i32 noundef signext %192, i32 noundef %193, i32 noundef %194, i32 noundef %195, i32 noundef %196, i32 noundef %197, i32 noundef %198, i32 noundef %199, i32 noundef %200)
  store i32 %call218, ptr %spad_rows, align 4
  %201 = load i32, ptr %stride.addr, align 4
  %202 = load i32, ptr %input_dilation.addr, align 4
  %203 = load i32, ptr %kernel_dilation.addr, align 4
  %204 = load i8, ptr %downsample, align 1
  %tobool219 = trunc i8 %204 to i1
  %205 = load i8, ptr %trans_weight_0132.addr, align 1
  %tobool220 = trunc i8 %205 to i1
  %206 = load i8, ptr %trans_input_3120.addr, align 1
  %tobool221 = trunc i8 %206 to i1
  %arrayidx222 = getelementptr inbounds [7 x i32], ptr %args_candidate185, i64 0, i64 0
  %207 = load i32, ptr %arrayidx222, align 4
  %arrayidx223 = getelementptr inbounds [7 x i32], ptr %args_candidate185, i64 0, i64 1
  %208 = load i32, ptr %arrayidx223, align 4
  %arrayidx224 = getelementptr inbounds [7 x i32], ptr %args_candidate185, i64 0, i64 2
  %209 = load i32, ptr %arrayidx224, align 4
  %arrayidx225 = getelementptr inbounds [7 x i32], ptr %args_candidate185, i64 0, i64 3
  %210 = load i32, ptr %arrayidx225, align 4
  %arrayidx226 = getelementptr inbounds [7 x i32], ptr %args_candidate185, i64 0, i64 4
  %211 = load i32, ptr %arrayidx226, align 4
  %arrayidx227 = getelementptr inbounds [7 x i32], ptr %args_candidate185, i64 0, i64 5
  %212 = load i32, ptr %arrayidx227, align 4
  %arrayidx228 = getelementptr inbounds [7 x i32], ptr %args_candidate185, i64 0, i64 6
  %213 = load i32, ptr %arrayidx228, align 4
  %214 = load i32, ptr %pool_size.addr, align 4
  %215 = load i32, ptr %pool_stride.addr, align 4
  %call229 = call signext i32 @tiled_conv_total_spad_rows(i1 noundef zeroext true, i32 noundef signext %201, i32 noundef signext %202, i32 noundef signext %203, i1 noundef zeroext %tobool219, i1 noundef zeroext %tobool220, i1 noundef zeroext %tobool221, i32 noundef signext %207, i32 noundef %208, i32 noundef %209, i32 noundef %210, i32 noundef %211, i32 noundef %212, i32 noundef %213, i32 noundef %214, i32 noundef %215)
  store i32 %call229, ptr %acc_rows, align 4
  %216 = load i32, ptr %spad_rows, align 4
  %cmp230 = icmp sle i32 %216, 8192
  br i1 %cmp230, label %land.lhs.true232, label %if.end238

land.lhs.true232:                                 ; preds = %if.end207
  %217 = load i32, ptr %acc_rows, align 4
  %cmp233 = icmp sle i32 %217, 512
  br i1 %cmp233, label %if.then235, label %if.end238

if.then235:                                       ; preds = %land.lhs.true232
  %218 = load i64, ptr %i180, align 8
  %arrayidx236 = getelementptr inbounds [7 x i32], ptr %args_candidate185, i64 0, i64 %218
  %219 = load i32, ptr %arrayidx236, align 4
  %220 = load i64, ptr %i180, align 8
  %arrayidx237 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 %220
  store i32 %219, ptr %arrayidx237, align 4
  store i8 0, ptr %nothing_increased, align 1
  br label %if.end238

if.end238:                                        ; preds = %if.then235, %land.lhs.true232, %if.end207
  br label %for.inc239

for.inc239:                                       ; preds = %if.end238, %if.then206
  %221 = load i64, ptr %i180, align 8
  %inc240 = add i64 %221, 1
  store i64 %inc240, ptr %i180, align 8
  br label %for.cond181, !llvm.loop !24

for.end241:                                       ; preds = %for.cond181
  br label %while.cond176, !llvm.loop !25

while.end242:                                     ; preds = %while.cond176
  %arrayidx243 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 0
  %222 = load i32, ptr %arrayidx243, align 4
  store i32 %222, ptr %batches, align 4
  %arrayidx244 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 1
  %223 = load i32, ptr %arrayidx244, align 4
  store i32 %223, ptr %orows, align 4
  %arrayidx245 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 2
  %224 = load i32, ptr %arrayidx245, align 4
  store i32 %224, ptr %ocols, align 4
  %arrayidx246 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 3
  %225 = load i32, ptr %arrayidx246, align 4
  store i32 %225, ptr %ochs, align 4
  %arrayidx247 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 4
  %226 = load i32, ptr %arrayidx247, align 4
  store i32 %226, ptr %krows, align 4
  %arrayidx248 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 5
  %227 = load i32, ptr %arrayidx248, align 4
  store i32 %227, ptr %kcols, align 4
  %arrayidx249 = getelementptr inbounds [7 x i32], ptr %args, i64 0, i64 6
  %228 = load i32, ptr %arrayidx249, align 4
  store i32 %228, ptr %kchs, align 4
  %229 = load i32, ptr %batch_size.addr, align 4
  %230 = load i32, ptr %in_dim.addr, align 4
  %231 = load i32, ptr %in_channels.addr, align 4
  %232 = load i32, ptr %out_channels.addr, align 4
  %233 = load i32, ptr %out_dim.addr, align 4
  %234 = load i32, ptr %stride.addr, align 4
  %235 = load i32, ptr %input_dilation.addr, align 4
  %236 = load i32, ptr %kernel_dilation.addr, align 4
  %237 = load i32, ptr %padding.addr, align 4
  %238 = load i32, ptr %kernel_dim.addr, align 4
  %239 = load i8, ptr %wrot180.addr, align 1
  %tobool250 = trunc i8 %239 to i1
  %240 = load i8, ptr %trans_output_1203.addr, align 1
  %tobool251 = trunc i8 %240 to i1
  %241 = load i8, ptr %trans_input_3120.addr, align 1
  %tobool252 = trunc i8 %241 to i1
  %242 = load i8, ptr %trans_weight_1203.addr, align 1
  %tobool253 = trunc i8 %242 to i1
  %243 = load i8, ptr %trans_weight_0132.addr, align 1
  %tobool254 = trunc i8 %243 to i1
  %244 = load i32, ptr %batches, align 4
  %245 = load i32, ptr %orows, align 4
  %246 = load i32, ptr %ocols, align 4
  %247 = load i32, ptr %ochs, align 4
  %248 = load i32, ptr %krows, align 4
  %249 = load i32, ptr %kcols, align 4
  %250 = load i32, ptr %kchs, align 4
  %251 = load ptr, ptr %input.addr, align 8
  %252 = load ptr, ptr %weights.addr, align 8
  %253 = load ptr, ptr %bias.addr, align 8
  %254 = load ptr, ptr %output.addr, align 8
  %255 = load i32, ptr %act.addr, align 4
  %256 = load float, ptr %scale.addr, align 4
  %257 = load i32, ptr %pool_size.addr, align 4
  %258 = load i8, ptr %no_pool, align 1
  %tobool255 = trunc i8 %258 to i1
  br i1 %tobool255, label %cond.true257, label %cond.false258

cond.true257:                                     ; preds = %while.end242
  br label %cond.end259

cond.false258:                                    ; preds = %while.end242
  %259 = load i32, ptr %pool_stride.addr, align 4
  br label %cond.end259

cond.end259:                                      ; preds = %cond.false258, %cond.true257
  %cond260 = phi i32 [ 0, %cond.true257 ], [ %259, %cond.false258 ]
  %260 = load i32, ptr %pool_padding.addr, align 4
  %261 = load i32, ptr %tiled_conv_type.addr, align 4
  call void @tiled_conv(i32 noundef signext %229, i32 noundef signext %230, i32 noundef signext %231, i32 noundef signext %232, i32 noundef signext %233, i32 noundef signext %234, i32 noundef signext %235, i32 noundef signext %236, i32 noundef %237, i32 noundef %238, i1 noundef %tobool250, i1 noundef %tobool251, i1 noundef %tobool252, i1 noundef %tobool253, i1 noundef %tobool254, i32 noundef %244, i32 noundef %245, i32 noundef %246, i32 noundef %247, i32 noundef %248, i32 noundef %249, i32 noundef %250, ptr noundef %251, ptr noundef %252, ptr noundef %253, ptr noundef %254, i32 noundef %255, float noundef %256, i32 noundef %257, i32 noundef %cond260, i32 noundef %260, i32 noundef %261)
  ret void
}

; Function Attrs: noinline nounwind optnone
define internal i64 @read_cycles() #0 {
entry:
  %cycles = alloca i64, align 8
  %0 = call i64 asm sideeffect "rdcycle $0", "=r"() #4, !srcloc !26
  store i64 %0, ptr %cycles, align 8
  %1 = load i64, ptr %cycles, align 8
  ret i64 %1
}

; Function Attrs: noinline nounwind optnone
define internal signext i32 @tiled_conv_total_spad_rows(i1 noundef zeroext %acc, i32 noundef signext %stride, i32 noundef signext %input_dilation, i32 noundef signext %kernel_dilation, i1 noundef zeroext %downsample, i1 noundef zeroext %trans_weight_0132, i1 noundef zeroext %trans_input_3120, i32 noundef signext %batches, i32 noundef %porows, i32 noundef %pocols, i32 noundef %ochs, i32 noundef %krows, i32 noundef %kcols, i32 noundef %kchs, i32 noundef %pool_size, i32 noundef %pool_stride) #0 {
entry:
  %acc.addr = alloca i8, align 1
  %stride.addr = alloca i32, align 4
  %input_dilation.addr = alloca i32, align 4
  %kernel_dilation.addr = alloca i32, align 4
  %downsample.addr = alloca i8, align 1
  %trans_weight_0132.addr = alloca i8, align 1
  %trans_input_3120.addr = alloca i8, align 1
  %batches.addr = alloca i32, align 4
  %porows.addr = alloca i32, align 4
  %pocols.addr = alloca i32, align 4
  %ochs.addr = alloca i32, align 4
  %krows.addr = alloca i32, align 4
  %kcols.addr = alloca i32, align 4
  %kchs.addr = alloca i32, align 4
  %pool_size.addr = alloca i32, align 4
  %pool_stride.addr = alloca i32, align 4
  %orows = alloca i32, align 4
  %ocols = alloca i32, align 4
  %krows_dilated = alloca i32, align 4
  %kcols_dilated = alloca i32, align 4
  %irows = alloca i32, align 4
  %icols = alloca i32, align 4
  %ichs = alloca i32, align 4
  %in_channels_per_bank = alloca i32, align 4
  %out_channels_per_bank = alloca i32, align 4
  %batches_per_bank = alloca i32, align 4
  %A_rows = alloca i32, align 4
  %B_rows = alloca i32, align 4
  %C_rows = alloca i32, align 4
  %frombool = zext i1 %acc to i8
  store i8 %frombool, ptr %acc.addr, align 1
  store i32 %stride, ptr %stride.addr, align 4
  store i32 %input_dilation, ptr %input_dilation.addr, align 4
  store i32 %kernel_dilation, ptr %kernel_dilation.addr, align 4
  %frombool1 = zext i1 %downsample to i8
  store i8 %frombool1, ptr %downsample.addr, align 1
  %frombool2 = zext i1 %trans_weight_0132 to i8
  store i8 %frombool2, ptr %trans_weight_0132.addr, align 1
  %frombool3 = zext i1 %trans_input_3120 to i8
  store i8 %frombool3, ptr %trans_input_3120.addr, align 1
  store i32 %batches, ptr %batches.addr, align 4
  store i32 %porows, ptr %porows.addr, align 4
  store i32 %pocols, ptr %pocols.addr, align 4
  store i32 %ochs, ptr %ochs.addr, align 4
  store i32 %krows, ptr %krows.addr, align 4
  store i32 %kcols, ptr %kcols.addr, align 4
  store i32 %kchs, ptr %kchs.addr, align 4
  store i32 %pool_size, ptr %pool_size.addr, align 4
  store i32 %pool_stride, ptr %pool_stride.addr, align 4
  %0 = load i32, ptr %porows.addr, align 4
  %1 = load i32, ptr %pool_stride.addr, align 4
  %mul = mul nsw i32 %0, %1
  %2 = load i32, ptr %pool_size.addr, align 4
  %add = add nsw i32 %mul, %2
  %sub = sub nsw i32 %add, 1
  store i32 %sub, ptr %orows, align 4
  %3 = load i32, ptr %pocols.addr, align 4
  %4 = load i32, ptr %pool_stride.addr, align 4
  %mul4 = mul nsw i32 %3, %4
  %5 = load i32, ptr %pool_size.addr, align 4
  %add5 = add nsw i32 %mul4, %5
  %sub6 = sub nsw i32 %add5, 1
  store i32 %sub6, ptr %ocols, align 4
  %6 = load i32, ptr %krows.addr, align 4
  %7 = load i32, ptr %kernel_dilation.addr, align 4
  %sub7 = sub nsw i32 %7, 1
  %8 = load i32, ptr %krows.addr, align 4
  %sub8 = sub nsw i32 %8, 1
  %mul9 = mul nsw i32 %sub7, %sub8
  %add10 = add nsw i32 %6, %mul9
  store i32 %add10, ptr %krows_dilated, align 4
  %9 = load i32, ptr %kcols.addr, align 4
  %10 = load i32, ptr %kernel_dilation.addr, align 4
  %sub11 = sub nsw i32 %10, 1
  %11 = load i32, ptr %kcols.addr, align 4
  %sub12 = sub nsw i32 %11, 1
  %mul13 = mul nsw i32 %sub11, %sub12
  %add14 = add nsw i32 %9, %mul13
  store i32 %add14, ptr %kcols_dilated, align 4
  %12 = load i32, ptr %orows, align 4
  %13 = load i32, ptr %stride.addr, align 4
  %mul15 = mul nsw i32 %12, %13
  %14 = load i32, ptr %krows_dilated, align 4
  %add16 = add nsw i32 %mul15, %14
  %sub17 = sub nsw i32 %add16, 1
  store i32 %sub17, ptr %irows, align 4
  %15 = load i32, ptr %ocols, align 4
  %16 = load i32, ptr %stride.addr, align 4
  %mul18 = mul nsw i32 %15, %16
  %17 = load i32, ptr %kcols_dilated, align 4
  %add19 = add nsw i32 %mul18, %17
  %sub20 = sub nsw i32 %add19, 1
  store i32 %sub20, ptr %icols, align 4
  %18 = load i32, ptr %kchs.addr, align 4
  store i32 %18, ptr %ichs, align 4
  %19 = load i32, ptr %irows, align 4
  %20 = load i32, ptr %input_dilation.addr, align 4
  %div = sdiv i32 %19, %20
  %21 = load i32, ptr %irows, align 4
  %22 = load i32, ptr %input_dilation.addr, align 4
  %rem = srem i32 %21, %22
  %cmp = icmp ne i32 %rem, 0
  %conv = zext i1 %cmp to i32
  %add21 = add nsw i32 %div, %conv
  store i32 %add21, ptr %irows, align 4
  %23 = load i32, ptr %icols, align 4
  %24 = load i32, ptr %input_dilation.addr, align 4
  %div22 = sdiv i32 %23, %24
  %25 = load i32, ptr %icols, align 4
  %26 = load i32, ptr %input_dilation.addr, align 4
  %rem23 = srem i32 %25, %26
  %cmp24 = icmp ne i32 %rem23, 0
  %conv25 = zext i1 %cmp24 to i32
  %add26 = add nsw i32 %div22, %conv25
  store i32 %add26, ptr %icols, align 4
  %27 = load i32, ptr %ichs, align 4
  %div27 = sdiv i32 %27, 16
  %28 = load i32, ptr %ichs, align 4
  %rem28 = srem i32 %28, 16
  %cmp29 = icmp ne i32 %rem28, 0
  %conv30 = zext i1 %cmp29 to i32
  %add31 = add nsw i32 %div27, %conv30
  store i32 %add31, ptr %in_channels_per_bank, align 4
  %29 = load i32, ptr %ochs.addr, align 4
  %div32 = sdiv i32 %29, 16
  %30 = load i32, ptr %ochs.addr, align 4
  %rem33 = srem i32 %30, 16
  %cmp34 = icmp ne i32 %rem33, 0
  %conv35 = zext i1 %cmp34 to i32
  %add36 = add nsw i32 %div32, %conv35
  store i32 %add36, ptr %out_channels_per_bank, align 4
  %31 = load i32, ptr %batches.addr, align 4
  %div37 = sdiv i32 %31, 16
  %32 = load i32, ptr %batches.addr, align 4
  %rem38 = srem i32 %32, 16
  %cmp39 = icmp ne i32 %rem38, 0
  %conv40 = zext i1 %cmp39 to i32
  %add41 = add nsw i32 %div37, %conv40
  store i32 %add41, ptr %batches_per_bank, align 4
  %33 = load i8, ptr %trans_input_3120.addr, align 1
  %tobool = trunc i8 %33 to i1
  br i1 %tobool, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  %34 = load i32, ptr %batches_per_bank, align 4
  %35 = load i32, ptr %ichs, align 4
  %mul43 = mul nsw i32 %34, %35
  %36 = load i32, ptr %irows, align 4
  %37 = load i8, ptr %downsample.addr, align 1
  %tobool44 = trunc i8 %37 to i1
  %conv45 = zext i1 %tobool44 to i32
  %shr = ashr i32 %36, %conv45
  %mul46 = mul nsw i32 %mul43, %shr
  %38 = load i32, ptr %icols, align 4
  %39 = load i8, ptr %downsample.addr, align 1
  %tobool47 = trunc i8 %39 to i1
  %conv48 = zext i1 %tobool47 to i32
  %shr49 = ashr i32 %38, %conv48
  %mul50 = mul nsw i32 %mul46, %shr49
  br label %cond.end

cond.false:                                       ; preds = %entry
  %40 = load i32, ptr %in_channels_per_bank, align 4
  %41 = load i32, ptr %batches.addr, align 4
  %mul51 = mul nsw i32 %40, %41
  %42 = load i32, ptr %irows, align 4
  %43 = load i8, ptr %downsample.addr, align 1
  %tobool52 = trunc i8 %43 to i1
  %conv53 = zext i1 %tobool52 to i32
  %shr54 = ashr i32 %42, %conv53
  %mul55 = mul nsw i32 %mul51, %shr54
  %44 = load i32, ptr %icols, align 4
  %45 = load i8, ptr %downsample.addr, align 1
  %tobool56 = trunc i8 %45 to i1
  %conv57 = zext i1 %tobool56 to i32
  %shr58 = ashr i32 %44, %conv57
  %mul59 = mul nsw i32 %mul55, %shr58
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %mul50, %cond.true ], [ %mul59, %cond.false ]
  store i32 %cond, ptr %A_rows, align 4
  %46 = load i8, ptr %trans_weight_0132.addr, align 1
  %tobool60 = trunc i8 %46 to i1
  br i1 %tobool60, label %cond.true62, label %cond.false66

cond.true62:                                      ; preds = %cond.end
  %47 = load i32, ptr %in_channels_per_bank, align 4
  %48 = load i32, ptr %kcols.addr, align 4
  %mul63 = mul nsw i32 %47, %48
  %49 = load i32, ptr %krows.addr, align 4
  %mul64 = mul nsw i32 %mul63, %49
  %50 = load i32, ptr %ochs.addr, align 4
  %mul65 = mul nsw i32 %mul64, %50
  br label %cond.end70

cond.false66:                                     ; preds = %cond.end
  %51 = load i32, ptr %out_channels_per_bank, align 4
  %52 = load i32, ptr %kcols.addr, align 4
  %mul67 = mul nsw i32 %51, %52
  %53 = load i32, ptr %krows.addr, align 4
  %mul68 = mul nsw i32 %mul67, %53
  %54 = load i32, ptr %kchs.addr, align 4
  %mul69 = mul nsw i32 %mul68, %54
  br label %cond.end70

cond.end70:                                       ; preds = %cond.false66, %cond.true62
  %cond71 = phi i32 [ %mul65, %cond.true62 ], [ %mul69, %cond.false66 ]
  store i32 %cond71, ptr %B_rows, align 4
  %55 = load i32, ptr %out_channels_per_bank, align 4
  %56 = load i32, ptr %batches.addr, align 4
  %mul72 = mul nsw i32 %55, %56
  %57 = load i32, ptr %orows, align 4
  %mul73 = mul nsw i32 %mul72, %57
  %58 = load i32, ptr %ocols, align 4
  %mul74 = mul nsw i32 %mul73, %58
  store i32 %mul74, ptr %C_rows, align 4
  %59 = load i8, ptr %acc.addr, align 1
  %tobool75 = trunc i8 %59 to i1
  br i1 %tobool75, label %cond.true77, label %cond.false78

cond.true77:                                      ; preds = %cond.end70
  %60 = load i32, ptr %C_rows, align 4
  br label %cond.end80

cond.false78:                                     ; preds = %cond.end70
  %61 = load i32, ptr %A_rows, align 4
  %62 = load i32, ptr %B_rows, align 4
  %add79 = add nsw i32 %61, %62
  br label %cond.end80

cond.end80:                                       ; preds = %cond.false78, %cond.true77
  %cond81 = phi i32 [ %60, %cond.true77 ], [ %add79, %cond.false78 ]
  ret i32 %cond81
}

; Function Attrs: noinline nounwind optnone
define internal void @tiled_conv(i32 noundef signext %batch_size, i32 noundef signext %in_dim, i32 noundef signext %in_channels, i32 noundef signext %out_channels, i32 noundef signext %out_dim, i32 noundef signext %stride, i32 noundef signext %input_dilation, i32 noundef signext %kernel_dilation, i32 noundef %padding, i32 noundef %kernel_dim, i1 noundef %wrot180, i1 noundef %trans_output_1203, i1 noundef %trans_input_3120, i1 noundef %trans_weight_1203, i1 noundef %trans_weight_0132, i32 noundef %batches, i32 noundef %porows, i32 noundef %pocols, i32 noundef %pochs, i32 noundef %krows, i32 noundef %kcols, i32 noundef %kchs, ptr noundef %input, ptr noundef %weights, ptr noundef %bias, ptr noundef %output, i32 noundef %act, float noundef %scale, i32 noundef %pool_size, i32 noundef %pool_stride, i32 noundef %pool_padding, i32 noundef %tiled_conv_type) #0 {
entry:
  %batch_size.addr = alloca i32, align 4
  %in_dim.addr = alloca i32, align 4
  %in_channels.addr = alloca i32, align 4
  %out_channels.addr = alloca i32, align 4
  %out_dim.addr = alloca i32, align 4
  %stride.addr = alloca i32, align 4
  %input_dilation.addr = alloca i32, align 4
  %kernel_dilation.addr = alloca i32, align 4
  %padding.addr = alloca i32, align 4
  %kernel_dim.addr = alloca i32, align 4
  %wrot180.addr = alloca i8, align 1
  %trans_output_1203.addr = alloca i8, align 1
  %trans_input_3120.addr = alloca i8, align 1
  %trans_weight_1203.addr = alloca i8, align 1
  %trans_weight_0132.addr = alloca i8, align 1
  %batches.addr = alloca i32, align 4
  %porows.addr = alloca i32, align 4
  %pocols.addr = alloca i32, align 4
  %pochs.addr = alloca i32, align 4
  %krows.addr = alloca i32, align 4
  %kcols.addr = alloca i32, align 4
  %kchs.addr = alloca i32, align 4
  %input.addr = alloca ptr, align 8
  %weights.addr = alloca ptr, align 8
  %bias.addr = alloca ptr, align 8
  %output.addr = alloca ptr, align 8
  %act.addr = alloca i32, align 4
  %scale.addr = alloca float, align 4
  %pool_size.addr = alloca i32, align 4
  %pool_stride.addr = alloca i32, align 4
  %pool_padding.addr = alloca i32, align 4
  %tiled_conv_type.addr = alloca i32, align 4
  %no_bias = alloca i8, align 1
  %no_pool = alloca i8, align 1
  %downsample = alloca i8, align 1
  %input_dilated = alloca i32, align 4
  %st_dram_stride = alloca i64, align 8
  %pool_out_dim = alloca i32, align 4
  %dilated_in_dim = alloca i32, align 4
  %b = alloca i32, align 4
  %porow = alloca i32, align 4
  %orow = alloca i32, align 4
  %pocol = alloca i32, align 4
  %ocol = alloca i32, align 4
  %poch = alloca i32, align 4
  %krow = alloca i32, align 4
  %orow_floored = alloca i32, align 4
  %irow = alloca i32, align 4
  %kcol = alloca i32, align 4
  %ocol_floored = alloca i32, align 4
  %icol = alloca i32, align 4
  %kch = alloca i32, align 4
  %out = alloca ptr, align 8
  %bias_ = alloca ptr, align 8
  %batches_ = alloca i32, align 4
  %porows_ = alloca i32, align 4
  %pocols_ = alloca i32, align 4
  %pochs_ = alloca i32, align 4
  %krows_ = alloca i32, align 4
  %kcols_ = alloca i32, align 4
  %kchs_ = alloca i32, align 4
  %ocols_ = alloca i32, align 4
  %orows_ = alloca i32, align 4
  %plpad = alloca i32, align 4
  %prpad = alloca i32, align 4
  %pupad = alloca i32, align 4
  %pdpad = alloca i32, align 4
  %dilated_krows_ = alloca i32, align 4
  %dilated_kcols_ = alloca i32, align 4
  %icols_ = alloca i32, align 4
  %irows_ = alloca i32, align 4
  %lpad = alloca i32, align 4
  %rpad = alloca i32, align 4
  %upad = alloca i32, align 4
  %dpad = alloca i32, align 4
  %krow_ = alloca i32, align 4
  %kcol_ = alloca i32, align 4
  %weights_slice = alloca ptr, align 8
  %in = alloca ptr, align 8
  store i32 %batch_size, ptr %batch_size.addr, align 4
  store i32 %in_dim, ptr %in_dim.addr, align 4
  store i32 %in_channels, ptr %in_channels.addr, align 4
  store i32 %out_channels, ptr %out_channels.addr, align 4
  store i32 %out_dim, ptr %out_dim.addr, align 4
  store i32 %stride, ptr %stride.addr, align 4
  store i32 %input_dilation, ptr %input_dilation.addr, align 4
  store i32 %kernel_dilation, ptr %kernel_dilation.addr, align 4
  store i32 %padding, ptr %padding.addr, align 4
  store i32 %kernel_dim, ptr %kernel_dim.addr, align 4
  %frombool = zext i1 %wrot180 to i8
  store i8 %frombool, ptr %wrot180.addr, align 1
  %frombool1 = zext i1 %trans_output_1203 to i8
  store i8 %frombool1, ptr %trans_output_1203.addr, align 1
  %frombool2 = zext i1 %trans_input_3120 to i8
  store i8 %frombool2, ptr %trans_input_3120.addr, align 1
  %frombool3 = zext i1 %trans_weight_1203 to i8
  store i8 %frombool3, ptr %trans_weight_1203.addr, align 1
  %frombool4 = zext i1 %trans_weight_0132 to i8
  store i8 %frombool4, ptr %trans_weight_0132.addr, align 1
  store i32 %batches, ptr %batches.addr, align 4
  store i32 %porows, ptr %porows.addr, align 4
  store i32 %pocols, ptr %pocols.addr, align 4
  store i32 %pochs, ptr %pochs.addr, align 4
  store i32 %krows, ptr %krows.addr, align 4
  store i32 %kcols, ptr %kcols.addr, align 4
  store i32 %kchs, ptr %kchs.addr, align 4
  store ptr %input, ptr %input.addr, align 8
  store ptr %weights, ptr %weights.addr, align 8
  store ptr %bias, ptr %bias.addr, align 8
  store ptr %output, ptr %output.addr, align 8
  store i32 %act, ptr %act.addr, align 4
  store float %scale, ptr %scale.addr, align 4
  store i32 %pool_size, ptr %pool_size.addr, align 4
  store i32 %pool_stride, ptr %pool_stride.addr, align 4
  store i32 %pool_padding, ptr %pool_padding.addr, align 4
  store i32 %tiled_conv_type, ptr %tiled_conv_type.addr, align 4
  %0 = load i32, ptr %batch_size.addr, align 4
  %1 = load i32, ptr %in_dim.addr, align 4
  %2 = load i32, ptr %in_channels.addr, align 4
  %3 = load i32, ptr %out_channels.addr, align 4
  %4 = load i32, ptr %out_dim.addr, align 4
  %5 = load i32, ptr %stride.addr, align 4
  %6 = load i32, ptr %input_dilation.addr, align 4
  %7 = load i32, ptr %kernel_dilation.addr, align 4
  %8 = load i32, ptr %padding.addr, align 4
  %9 = load i32, ptr %kernel_dim.addr, align 4
  %10 = load i32, ptr %batches.addr, align 4
  %11 = load i32, ptr %porows.addr, align 4
  %12 = load i32, ptr %pochs.addr, align 4
  %13 = load i32, ptr %krows.addr, align 4
  %14 = load i32, ptr %kcols.addr, align 4
  %15 = load i32, ptr %kchs.addr, align 4
  %call = call signext i32 (ptr, ...) @printf(ptr noundef @.str.15, i32 noundef signext %0, i32 noundef signext %1, i32 noundef signext %2, i32 noundef signext %3, i32 noundef signext %4, i32 noundef signext %5, i32 noundef signext %6, i32 noundef %7, i32 noundef %8, i32 noundef %9, i32 noundef %10, i32 noundef %11, i32 noundef %12, i32 noundef %13, i32 noundef %14, i32 noundef %15)
  store i8 0, ptr %no_bias, align 1
  %16 = load ptr, ptr %bias.addr, align 8
  %cmp = icmp eq ptr %16, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store ptr inttoptr (i64 1 to ptr), ptr %bias.addr, align 8
  store i8 1, ptr %no_bias, align 1
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %17 = load i32, ptr %pool_stride.addr, align 4
  %cmp5 = icmp eq i32 %17, 0
  %frombool6 = zext i1 %cmp5 to i8
  store i8 %frombool6, ptr %no_pool, align 1
  %18 = load i8, ptr %no_pool, align 1
  %tobool = trunc i8 %18 to i1
  br i1 %tobool, label %if.then7, label %if.end8

if.then7:                                         ; preds = %if.end
  store i32 1, ptr %pool_size.addr, align 4
  store i32 1, ptr %pool_stride.addr, align 4
  store i32 0, ptr %pool_padding.addr, align 4
  br label %if.end8

if.end8:                                          ; preds = %if.then7, %if.end
  %19 = load i32, ptr %stride.addr, align 4
  %cmp9 = icmp eq i32 %19, 2
  br i1 %cmp9, label %land.lhs.true, label %land.end

land.lhs.true:                                    ; preds = %if.end8
  %20 = load i32, ptr %kernel_dim.addr, align 4
  %cmp10 = icmp eq i32 %20, 1
  br i1 %cmp10, label %land.lhs.true11, label %land.end

land.lhs.true11:                                  ; preds = %land.lhs.true
  %21 = load i32, ptr %in_dim.addr, align 4
  %rem = srem i32 %21, 2
  %cmp12 = icmp eq i32 %rem, 0
  br i1 %cmp12, label %land.lhs.true13, label %land.end

land.lhs.true13:                                  ; preds = %land.lhs.true11
  %22 = load i32, ptr %padding.addr, align 4
  %cmp14 = icmp eq i32 %22, 0
  br i1 %cmp14, label %land.lhs.true15, label %land.end

land.lhs.true15:                                  ; preds = %land.lhs.true13
  %23 = load i8, ptr %no_pool, align 1
  %tobool16 = trunc i8 %23 to i1
  br i1 %tobool16, label %land.lhs.true17, label %land.end

land.lhs.true17:                                  ; preds = %land.lhs.true15
  %24 = load i32, ptr %input_dilation.addr, align 4
  %cmp18 = icmp eq i32 %24, 1
  br i1 %cmp18, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %land.lhs.true17
  %25 = load i8, ptr %trans_input_3120.addr, align 1
  %tobool19 = trunc i8 %25 to i1
  %lnot = xor i1 %tobool19, true
  br label %land.end

land.end:                                         ; preds = %land.rhs, %land.lhs.true17, %land.lhs.true15, %land.lhs.true13, %land.lhs.true11, %land.lhs.true, %if.end8
  %26 = phi i1 [ false, %land.lhs.true17 ], [ false, %land.lhs.true15 ], [ false, %land.lhs.true13 ], [ false, %land.lhs.true11 ], [ false, %land.lhs.true ], [ false, %if.end8 ], [ %lnot, %land.rhs ]
  %frombool20 = zext i1 %26 to i8
  store i8 %frombool20, ptr %downsample, align 1
  %27 = load i32, ptr %input_dilation.addr, align 4
  %cmp21 = icmp eq i32 %27, 2
  %conv = zext i1 %cmp21 to i32
  store i32 %conv, ptr %input_dilated, align 4
  %28 = load i8, ptr %trans_output_1203.addr, align 1
  %tobool22 = trunc i8 %28 to i1
  br i1 %tobool22, label %cond.true, label %cond.false

cond.true:                                        ; preds = %land.end
  %29 = load i32, ptr %batch_size.addr, align 4
  %30 = load i32, ptr %out_channels.addr, align 4
  %mul = mul nsw i32 %29, %30
  %conv24 = sext i32 %mul to i64
  %mul25 = mul i64 %conv24, 1
  br label %cond.end

cond.false:                                       ; preds = %land.end
  %31 = load i32, ptr %out_channels.addr, align 4
  %conv26 = sext i32 %31 to i64
  %mul27 = mul i64 %conv26, 1
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i64 [ %mul25, %cond.true ], [ %mul27, %cond.false ]
  store i64 %cond, ptr %st_dram_stride, align 8
  call void @llvm.riscv.configSt(i64 2, i64 4575657221408423955)
  %call28 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.16)
  call void @llvm.riscv.configEx(i64 131076, i64 281474976710656)
  %call29 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.17)
  %32 = load i32, ptr %out_dim.addr, align 4
  %33 = load i32, ptr %pool_padding.addr, align 4
  %mul30 = mul nsw i32 2, %33
  %add = add nsw i32 %32, %mul30
  %34 = load i32, ptr %pool_size.addr, align 4
  %sub = sub nsw i32 %add, %34
  %35 = load i32, ptr %pool_stride.addr, align 4
  %div = sdiv i32 %sub, %35
  %add31 = add nsw i32 %div, 1
  store i32 %add31, ptr %pool_out_dim, align 4
  %36 = load i32, ptr %in_dim.addr, align 4
  %37 = load i32, ptr %input_dilation.addr, align 4
  %sub32 = sub nsw i32 %37, 1
  %38 = load i32, ptr %in_dim.addr, align 4
  %sub33 = sub nsw i32 %38, 1
  %mul34 = mul nsw i32 %sub32, %sub33
  %add35 = add nsw i32 %36, %mul34
  store i32 %add35, ptr %dilated_in_dim, align 4
  store i32 0, ptr %b, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc413, %cond.end
  %39 = load i32, ptr %b, align 4
  %40 = load i32, ptr %batch_size.addr, align 4
  %cmp36 = icmp slt i32 %39, %40
  br i1 %cmp36, label %for.body, label %for.end415

for.body:                                         ; preds = %for.cond
  store i32 0, ptr %porow, align 4
  br label %for.cond38

for.cond38:                                       ; preds = %for.inc410, %for.body
  %41 = load i32, ptr %porow, align 4
  %42 = load i32, ptr %pool_out_dim, align 4
  %cmp39 = icmp slt i32 %41, %42
  br i1 %cmp39, label %for.body41, label %for.end412

for.body41:                                       ; preds = %for.cond38
  %43 = load i32, ptr %porow, align 4
  %44 = load i32, ptr %pool_stride.addr, align 4
  %mul42 = mul nsw i32 %43, %44
  %45 = load i32, ptr %pool_padding.addr, align 4
  %sub43 = sub nsw i32 %mul42, %45
  store i32 %sub43, ptr %orow, align 4
  store i32 0, ptr %pocol, align 4
  br label %for.cond44

for.cond44:                                       ; preds = %for.inc407, %for.body41
  %46 = load i32, ptr %pocol, align 4
  %47 = load i32, ptr %pool_out_dim, align 4
  %cmp45 = icmp slt i32 %46, %47
  br i1 %cmp45, label %for.body47, label %for.end409

for.body47:                                       ; preds = %for.cond44
  %48 = load i32, ptr %pocol, align 4
  %49 = load i32, ptr %pool_stride.addr, align 4
  %mul48 = mul nsw i32 %48, %49
  %50 = load i32, ptr %pool_padding.addr, align 4
  %sub49 = sub nsw i32 %mul48, %50
  store i32 %sub49, ptr %ocol, align 4
  store i32 0, ptr %poch, align 4
  br label %for.cond50

for.cond50:                                       ; preds = %for.inc404, %for.body47
  %51 = load i32, ptr %poch, align 4
  %52 = load i32, ptr %out_channels.addr, align 4
  %cmp51 = icmp slt i32 %51, %52
  br i1 %cmp51, label %for.body53, label %for.end406

for.body53:                                       ; preds = %for.cond50
  store i32 0, ptr %krow, align 4
  br label %for.cond54

for.cond54:                                       ; preds = %for.inc401, %for.body53
  %53 = load i32, ptr %krow, align 4
  %54 = load i32, ptr %kernel_dim.addr, align 4
  %cmp55 = icmp slt i32 %53, %54
  br i1 %cmp55, label %for.body57, label %for.end403

for.body57:                                       ; preds = %for.cond54
  %55 = load i32, ptr %orow, align 4
  %cmp58 = icmp slt i32 %55, 0
  br i1 %cmp58, label %cond.true60, label %cond.false61

cond.true60:                                      ; preds = %for.body57
  br label %cond.end62

cond.false61:                                     ; preds = %for.body57
  %56 = load i32, ptr %orow, align 4
  br label %cond.end62

cond.end62:                                       ; preds = %cond.false61, %cond.true60
  %cond63 = phi i32 [ 0, %cond.true60 ], [ %56, %cond.false61 ]
  store i32 %cond63, ptr %orow_floored, align 4
  %57 = load i32, ptr %orow_floored, align 4
  %58 = load i32, ptr %stride.addr, align 4
  %mul64 = mul nsw i32 %57, %58
  %59 = load i32, ptr %krow, align 4
  %60 = load i32, ptr %kernel_dilation.addr, align 4
  %mul65 = mul nsw i32 %59, %60
  %add66 = add nsw i32 %mul64, %mul65
  %61 = load i32, ptr %padding.addr, align 4
  %sub67 = sub nsw i32 %add66, %61
  store i32 %sub67, ptr %irow, align 4
  store i32 0, ptr %kcol, align 4
  br label %for.cond68

for.cond68:                                       ; preds = %for.inc398, %cond.end62
  %62 = load i32, ptr %kcol, align 4
  %63 = load i32, ptr %kernel_dim.addr, align 4
  %cmp69 = icmp slt i32 %62, %63
  br i1 %cmp69, label %for.body71, label %for.end400

for.body71:                                       ; preds = %for.cond68
  %64 = load i32, ptr %ocol, align 4
  %cmp72 = icmp slt i32 %64, 0
  br i1 %cmp72, label %cond.true74, label %cond.false75

cond.true74:                                      ; preds = %for.body71
  br label %cond.end76

cond.false75:                                     ; preds = %for.body71
  %65 = load i32, ptr %ocol, align 4
  br label %cond.end76

cond.end76:                                       ; preds = %cond.false75, %cond.true74
  %cond77 = phi i32 [ 0, %cond.true74 ], [ %65, %cond.false75 ]
  store i32 %cond77, ptr %ocol_floored, align 4
  %66 = load i32, ptr %ocol_floored, align 4
  %67 = load i32, ptr %stride.addr, align 4
  %mul78 = mul nsw i32 %66, %67
  %68 = load i32, ptr %kcol, align 4
  %69 = load i32, ptr %kernel_dilation.addr, align 4
  %mul79 = mul nsw i32 %68, %69
  %add80 = add nsw i32 %mul78, %mul79
  %70 = load i32, ptr %padding.addr, align 4
  %sub81 = sub nsw i32 %add80, %70
  store i32 %sub81, ptr %icol, align 4
  store i32 0, ptr %kch, align 4
  br label %for.cond82

for.cond82:                                       ; preds = %for.inc, %cond.end76
  %71 = load i32, ptr %kch, align 4
  %72 = load i32, ptr %in_channels.addr, align 4
  %cmp83 = icmp slt i32 %71, %72
  br i1 %cmp83, label %for.body85, label %for.end

for.body85:                                       ; preds = %for.cond82
  %73 = load ptr, ptr %output.addr, align 8
  %74 = load i32, ptr %b, align 4
  %75 = load i32, ptr %pool_out_dim, align 4
  %mul86 = mul nsw i32 %74, %75
  %76 = load i32, ptr %pool_out_dim, align 4
  %mul87 = mul nsw i32 %mul86, %76
  %77 = load i32, ptr %porow, align 4
  %78 = load i32, ptr %pool_out_dim, align 4
  %mul88 = mul nsw i32 %77, %78
  %add89 = add nsw i32 %mul87, %mul88
  %79 = load i32, ptr %pocol, align 4
  %add90 = add nsw i32 %add89, %79
  %80 = load i32, ptr %out_channels.addr, align 4
  %mul91 = mul nsw i32 %add90, %80
  %idx.ext = sext i32 %mul91 to i64
  %add.ptr = getelementptr inbounds i8, ptr %73, i64 %idx.ext
  %81 = load i32, ptr %poch, align 4
  %idx.ext92 = sext i32 %81 to i64
  %add.ptr93 = getelementptr inbounds i8, ptr %add.ptr, i64 %idx.ext92
  store ptr %add.ptr93, ptr %out, align 8
  %82 = load i8, ptr %trans_output_1203.addr, align 1
  %tobool94 = trunc i8 %82 to i1
  br i1 %tobool94, label %if.then95, label %if.end106

if.then95:                                        ; preds = %for.body85
  %83 = load ptr, ptr %output.addr, align 8
  %84 = load i32, ptr %porow, align 4
  %85 = load i32, ptr %pool_out_dim, align 4
  %mul96 = mul nsw i32 %84, %85
  %86 = load i32, ptr %batch_size.addr, align 4
  %mul97 = mul nsw i32 %mul96, %86
  %87 = load i32, ptr %pocol, align 4
  %88 = load i32, ptr %batch_size.addr, align 4
  %mul98 = mul nsw i32 %87, %88
  %add99 = add nsw i32 %mul97, %mul98
  %89 = load i32, ptr %b, align 4
  %add100 = add nsw i32 %add99, %89
  %90 = load i32, ptr %out_channels.addr, align 4
  %mul101 = mul nsw i32 %add100, %90
  %idx.ext102 = sext i32 %mul101 to i64
  %add.ptr103 = getelementptr inbounds i8, ptr %83, i64 %idx.ext102
  %91 = load i32, ptr %poch, align 4
  %idx.ext104 = sext i32 %91 to i64
  %add.ptr105 = getelementptr inbounds i8, ptr %add.ptr103, i64 %idx.ext104
  store ptr %add.ptr105, ptr %out, align 8
  br label %if.end106

if.end106:                                        ; preds = %if.then95, %for.body85
  %92 = load i32, ptr %krow, align 4
  %93 = load i32, ptr %krows.addr, align 4
  %add107 = add nsw i32 %92, %93
  %94 = load i32, ptr %kernel_dim.addr, align 4
  %cmp108 = icmp slt i32 %add107, %94
  br i1 %cmp108, label %if.then117, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %if.end106
  %95 = load i32, ptr %kcol, align 4
  %96 = load i32, ptr %kcols.addr, align 4
  %add110 = add nsw i32 %95, %96
  %97 = load i32, ptr %kernel_dim.addr, align 4
  %cmp111 = icmp slt i32 %add110, %97
  br i1 %cmp111, label %if.then117, label %lor.lhs.false113

lor.lhs.false113:                                 ; preds = %lor.lhs.false
  %98 = load i32, ptr %kch, align 4
  %99 = load i32, ptr %kchs.addr, align 4
  %add114 = add nsw i32 %98, %99
  %100 = load i32, ptr %in_channels.addr, align 4
  %cmp115 = icmp slt i32 %add114, %100
  br i1 %cmp115, label %if.then117, label %if.end118

if.then117:                                       ; preds = %lor.lhs.false113, %lor.lhs.false, %if.end106
  store ptr null, ptr %out, align 8
  br label %if.end118

if.end118:                                        ; preds = %if.then117, %lor.lhs.false113
  %101 = load ptr, ptr %bias.addr, align 8
  %102 = load i32, ptr %poch, align 4
  %idx.ext119 = sext i32 %102 to i64
  %add.ptr120 = getelementptr inbounds i32, ptr %101, i64 %idx.ext119
  store ptr %add.ptr120, ptr %bias_, align 8
  %103 = load i32, ptr %krow, align 4
  %cmp121 = icmp sgt i32 %103, 0
  br i1 %cmp121, label %if.then129, label %lor.lhs.false123

lor.lhs.false123:                                 ; preds = %if.end118
  %104 = load i32, ptr %kcol, align 4
  %cmp124 = icmp sgt i32 %104, 0
  br i1 %cmp124, label %if.then129, label %lor.lhs.false126

lor.lhs.false126:                                 ; preds = %lor.lhs.false123
  %105 = load i32, ptr %kch, align 4
  %cmp127 = icmp sgt i32 %105, 0
  br i1 %cmp127, label %if.then129, label %if.end130

if.then129:                                       ; preds = %lor.lhs.false126, %lor.lhs.false123, %if.end118
  store ptr null, ptr %bias_, align 8
  br label %if.end130

if.end130:                                        ; preds = %if.then129, %lor.lhs.false126
  %106 = load i32, ptr %batch_size.addr, align 4
  %107 = load i32, ptr %b, align 4
  %sub131 = sub nsw i32 %106, %107
  %108 = load i32, ptr %batches.addr, align 4
  %cmp132 = icmp sgt i32 %sub131, %108
  br i1 %cmp132, label %cond.true134, label %cond.false135

cond.true134:                                     ; preds = %if.end130
  %109 = load i32, ptr %batches.addr, align 4
  br label %cond.end137

cond.false135:                                    ; preds = %if.end130
  %110 = load i32, ptr %batch_size.addr, align 4
  %111 = load i32, ptr %b, align 4
  %sub136 = sub nsw i32 %110, %111
  br label %cond.end137

cond.end137:                                      ; preds = %cond.false135, %cond.true134
  %cond138 = phi i32 [ %109, %cond.true134 ], [ %sub136, %cond.false135 ]
  store i32 %cond138, ptr %batches_, align 4
  %112 = load i32, ptr %pool_out_dim, align 4
  %113 = load i32, ptr %porow, align 4
  %sub139 = sub nsw i32 %112, %113
  %114 = load i32, ptr %porows.addr, align 4
  %cmp140 = icmp sgt i32 %sub139, %114
  br i1 %cmp140, label %cond.true142, label %cond.false143

cond.true142:                                     ; preds = %cond.end137
  %115 = load i32, ptr %porows.addr, align 4
  br label %cond.end145

cond.false143:                                    ; preds = %cond.end137
  %116 = load i32, ptr %pool_out_dim, align 4
  %117 = load i32, ptr %porow, align 4
  %sub144 = sub nsw i32 %116, %117
  br label %cond.end145

cond.end145:                                      ; preds = %cond.false143, %cond.true142
  %cond146 = phi i32 [ %115, %cond.true142 ], [ %sub144, %cond.false143 ]
  store i32 %cond146, ptr %porows_, align 4
  %118 = load i32, ptr %pool_out_dim, align 4
  %119 = load i32, ptr %pocol, align 4
  %sub147 = sub nsw i32 %118, %119
  %120 = load i32, ptr %pocols.addr, align 4
  %cmp148 = icmp sgt i32 %sub147, %120
  br i1 %cmp148, label %cond.true150, label %cond.false151

cond.true150:                                     ; preds = %cond.end145
  %121 = load i32, ptr %pocols.addr, align 4
  br label %cond.end153

cond.false151:                                    ; preds = %cond.end145
  %122 = load i32, ptr %pool_out_dim, align 4
  %123 = load i32, ptr %pocol, align 4
  %sub152 = sub nsw i32 %122, %123
  br label %cond.end153

cond.end153:                                      ; preds = %cond.false151, %cond.true150
  %cond154 = phi i32 [ %121, %cond.true150 ], [ %sub152, %cond.false151 ]
  store i32 %cond154, ptr %pocols_, align 4
  %124 = load i32, ptr %out_channels.addr, align 4
  %125 = load i32, ptr %poch, align 4
  %sub155 = sub nsw i32 %124, %125
  %126 = load i32, ptr %pochs.addr, align 4
  %cmp156 = icmp sgt i32 %sub155, %126
  br i1 %cmp156, label %cond.true158, label %cond.false159

cond.true158:                                     ; preds = %cond.end153
  %127 = load i32, ptr %pochs.addr, align 4
  br label %cond.end161

cond.false159:                                    ; preds = %cond.end153
  %128 = load i32, ptr %out_channels.addr, align 4
  %129 = load i32, ptr %poch, align 4
  %sub160 = sub nsw i32 %128, %129
  br label %cond.end161

cond.end161:                                      ; preds = %cond.false159, %cond.true158
  %cond162 = phi i32 [ %127, %cond.true158 ], [ %sub160, %cond.false159 ]
  store i32 %cond162, ptr %pochs_, align 4
  %130 = load i32, ptr %kernel_dim.addr, align 4
  %131 = load i32, ptr %krow, align 4
  %sub163 = sub nsw i32 %130, %131
  %132 = load i32, ptr %krows.addr, align 4
  %cmp164 = icmp sgt i32 %sub163, %132
  br i1 %cmp164, label %cond.true166, label %cond.false167

cond.true166:                                     ; preds = %cond.end161
  %133 = load i32, ptr %krows.addr, align 4
  br label %cond.end169

cond.false167:                                    ; preds = %cond.end161
  %134 = load i32, ptr %kernel_dim.addr, align 4
  %135 = load i32, ptr %krow, align 4
  %sub168 = sub nsw i32 %134, %135
  br label %cond.end169

cond.end169:                                      ; preds = %cond.false167, %cond.true166
  %cond170 = phi i32 [ %133, %cond.true166 ], [ %sub168, %cond.false167 ]
  store i32 %cond170, ptr %krows_, align 4
  %136 = load i32, ptr %kernel_dim.addr, align 4
  %137 = load i32, ptr %kcol, align 4
  %sub171 = sub nsw i32 %136, %137
  %138 = load i32, ptr %kcols.addr, align 4
  %cmp172 = icmp sgt i32 %sub171, %138
  br i1 %cmp172, label %cond.true174, label %cond.false175

cond.true174:                                     ; preds = %cond.end169
  %139 = load i32, ptr %kcols.addr, align 4
  br label %cond.end177

cond.false175:                                    ; preds = %cond.end169
  %140 = load i32, ptr %kernel_dim.addr, align 4
  %141 = load i32, ptr %kcol, align 4
  %sub176 = sub nsw i32 %140, %141
  br label %cond.end177

cond.end177:                                      ; preds = %cond.false175, %cond.true174
  %cond178 = phi i32 [ %139, %cond.true174 ], [ %sub176, %cond.false175 ]
  store i32 %cond178, ptr %kcols_, align 4
  %142 = load i32, ptr %in_channels.addr, align 4
  %143 = load i32, ptr %kch, align 4
  %sub179 = sub nsw i32 %142, %143
  %144 = load i32, ptr %kchs.addr, align 4
  %cmp180 = icmp sgt i32 %sub179, %144
  br i1 %cmp180, label %cond.true182, label %cond.false183

cond.true182:                                     ; preds = %cond.end177
  %145 = load i32, ptr %kchs.addr, align 4
  br label %cond.end185

cond.false183:                                    ; preds = %cond.end177
  %146 = load i32, ptr %in_channels.addr, align 4
  %147 = load i32, ptr %kch, align 4
  %sub184 = sub nsw i32 %146, %147
  br label %cond.end185

cond.end185:                                      ; preds = %cond.false183, %cond.true182
  %cond186 = phi i32 [ %145, %cond.true182 ], [ %sub184, %cond.false183 ]
  store i32 %cond186, ptr %kchs_, align 4
  %148 = load i32, ptr %pocols_, align 4
  %149 = load i32, ptr %pool_stride.addr, align 4
  %mul187 = mul nsw i32 %148, %149
  %150 = load i32, ptr %pool_size.addr, align 4
  %add188 = add nsw i32 %mul187, %150
  %sub189 = sub nsw i32 %add188, 1
  store i32 %sub189, ptr %ocols_, align 4
  %151 = load i32, ptr %porows_, align 4
  %152 = load i32, ptr %pool_stride.addr, align 4
  %mul190 = mul nsw i32 %151, %152
  %153 = load i32, ptr %pool_size.addr, align 4
  %add191 = add nsw i32 %mul190, %153
  %sub192 = sub nsw i32 %add191, 1
  store i32 %sub192, ptr %orows_, align 4
  %154 = load i32, ptr %ocol, align 4
  %cmp193 = icmp slt i32 %154, 0
  br i1 %cmp193, label %cond.true195, label %cond.false197

cond.true195:                                     ; preds = %cond.end185
  %155 = load i32, ptr %ocol, align 4
  %sub196 = sub nsw i32 0, %155
  br label %cond.end198

cond.false197:                                    ; preds = %cond.end185
  br label %cond.end198

cond.end198:                                      ; preds = %cond.false197, %cond.true195
  %cond199 = phi i32 [ %sub196, %cond.true195 ], [ 0, %cond.false197 ]
  store i32 %cond199, ptr %plpad, align 4
  %156 = load i32, ptr %ocol, align 4
  %157 = load i32, ptr %ocols_, align 4
  %add200 = add nsw i32 %156, %157
  %158 = load i32, ptr %out_dim.addr, align 4
  %cmp201 = icmp sgt i32 %add200, %158
  br i1 %cmp201, label %cond.true203, label %cond.false206

cond.true203:                                     ; preds = %cond.end198
  %159 = load i32, ptr %ocol, align 4
  %160 = load i32, ptr %ocols_, align 4
  %add204 = add nsw i32 %159, %160
  %161 = load i32, ptr %out_dim.addr, align 4
  %sub205 = sub nsw i32 %add204, %161
  br label %cond.end207

cond.false206:                                    ; preds = %cond.end198
  br label %cond.end207

cond.end207:                                      ; preds = %cond.false206, %cond.true203
  %cond208 = phi i32 [ %sub205, %cond.true203 ], [ 0, %cond.false206 ]
  store i32 %cond208, ptr %prpad, align 4
  %162 = load i32, ptr %orow, align 4
  %cmp209 = icmp slt i32 %162, 0
  br i1 %cmp209, label %cond.true211, label %cond.false213

cond.true211:                                     ; preds = %cond.end207
  %163 = load i32, ptr %orow, align 4
  %sub212 = sub nsw i32 0, %163
  br label %cond.end214

cond.false213:                                    ; preds = %cond.end207
  br label %cond.end214

cond.end214:                                      ; preds = %cond.false213, %cond.true211
  %cond215 = phi i32 [ %sub212, %cond.true211 ], [ 0, %cond.false213 ]
  store i32 %cond215, ptr %pupad, align 4
  %164 = load i32, ptr %orow, align 4
  %165 = load i32, ptr %orows_, align 4
  %add216 = add nsw i32 %164, %165
  %166 = load i32, ptr %out_dim.addr, align 4
  %cmp217 = icmp sgt i32 %add216, %166
  br i1 %cmp217, label %cond.true219, label %cond.false222

cond.true219:                                     ; preds = %cond.end214
  %167 = load i32, ptr %orow, align 4
  %168 = load i32, ptr %orows_, align 4
  %add220 = add nsw i32 %167, %168
  %169 = load i32, ptr %out_dim.addr, align 4
  %sub221 = sub nsw i32 %add220, %169
  br label %cond.end223

cond.false222:                                    ; preds = %cond.end214
  br label %cond.end223

cond.end223:                                      ; preds = %cond.false222, %cond.true219
  %cond224 = phi i32 [ %sub221, %cond.true219 ], [ 0, %cond.false222 ]
  store i32 %cond224, ptr %pdpad, align 4
  %170 = load i32, ptr %krows_, align 4
  %171 = load i32, ptr %kernel_dilation.addr, align 4
  %sub225 = sub nsw i32 %171, 1
  %172 = load i32, ptr %krows_, align 4
  %sub226 = sub nsw i32 %172, 1
  %mul227 = mul nsw i32 %sub225, %sub226
  %add228 = add nsw i32 %170, %mul227
  store i32 %add228, ptr %dilated_krows_, align 4
  %173 = load i32, ptr %kcols_, align 4
  %174 = load i32, ptr %kernel_dilation.addr, align 4
  %sub229 = sub nsw i32 %174, 1
  %175 = load i32, ptr %kcols_, align 4
  %sub230 = sub nsw i32 %175, 1
  %mul231 = mul nsw i32 %sub229, %sub230
  %add232 = add nsw i32 %173, %mul231
  store i32 %add232, ptr %dilated_kcols_, align 4
  %176 = load i32, ptr %ocols_, align 4
  %177 = load i32, ptr %plpad, align 4
  %sub233 = sub nsw i32 %176, %177
  %178 = load i32, ptr %prpad, align 4
  %sub234 = sub nsw i32 %sub233, %178
  %179 = load i32, ptr %stride.addr, align 4
  %mul235 = mul nsw i32 %sub234, %179
  %180 = load i32, ptr %dilated_kcols_, align 4
  %add236 = add nsw i32 %mul235, %180
  %sub237 = sub nsw i32 %add236, 1
  store i32 %sub237, ptr %icols_, align 4
  %181 = load i32, ptr %orows_, align 4
  %182 = load i32, ptr %pupad, align 4
  %sub238 = sub nsw i32 %181, %182
  %183 = load i32, ptr %pdpad, align 4
  %sub239 = sub nsw i32 %sub238, %183
  %184 = load i32, ptr %stride.addr, align 4
  %mul240 = mul nsw i32 %sub239, %184
  %185 = load i32, ptr %dilated_krows_, align 4
  %add241 = add nsw i32 %mul240, %185
  %sub242 = sub nsw i32 %add241, 1
  store i32 %sub242, ptr %irows_, align 4
  %186 = load i32, ptr %icol, align 4
  %cmp243 = icmp slt i32 %186, 0
  br i1 %cmp243, label %cond.true245, label %cond.false247

cond.true245:                                     ; preds = %cond.end223
  %187 = load i32, ptr %icol, align 4
  %sub246 = sub nsw i32 0, %187
  br label %cond.end248

cond.false247:                                    ; preds = %cond.end223
  br label %cond.end248

cond.end248:                                      ; preds = %cond.false247, %cond.true245
  %cond249 = phi i32 [ %sub246, %cond.true245 ], [ 0, %cond.false247 ]
  store i32 %cond249, ptr %lpad, align 4
  %188 = load i32, ptr %icol, align 4
  %189 = load i32, ptr %icols_, align 4
  %add250 = add nsw i32 %188, %189
  %190 = load i32, ptr %dilated_in_dim, align 4
  %cmp251 = icmp sgt i32 %add250, %190
  br i1 %cmp251, label %cond.true253, label %cond.false256

cond.true253:                                     ; preds = %cond.end248
  %191 = load i32, ptr %icol, align 4
  %192 = load i32, ptr %icols_, align 4
  %add254 = add nsw i32 %191, %192
  %193 = load i32, ptr %dilated_in_dim, align 4
  %sub255 = sub nsw i32 %add254, %193
  br label %cond.end257

cond.false256:                                    ; preds = %cond.end248
  br label %cond.end257

cond.end257:                                      ; preds = %cond.false256, %cond.true253
  %cond258 = phi i32 [ %sub255, %cond.true253 ], [ 0, %cond.false256 ]
  store i32 %cond258, ptr %rpad, align 4
  %194 = load i32, ptr %irow, align 4
  %cmp259 = icmp slt i32 %194, 0
  br i1 %cmp259, label %cond.true261, label %cond.false263

cond.true261:                                     ; preds = %cond.end257
  %195 = load i32, ptr %irow, align 4
  %sub262 = sub nsw i32 0, %195
  br label %cond.end264

cond.false263:                                    ; preds = %cond.end257
  br label %cond.end264

cond.end264:                                      ; preds = %cond.false263, %cond.true261
  %cond265 = phi i32 [ %sub262, %cond.true261 ], [ 0, %cond.false263 ]
  store i32 %cond265, ptr %upad, align 4
  %196 = load i32, ptr %irow, align 4
  %197 = load i32, ptr %irows_, align 4
  %add266 = add nsw i32 %196, %197
  %198 = load i32, ptr %dilated_in_dim, align 4
  %cmp267 = icmp sgt i32 %add266, %198
  br i1 %cmp267, label %cond.true269, label %cond.false272

cond.true269:                                     ; preds = %cond.end264
  %199 = load i32, ptr %irow, align 4
  %200 = load i32, ptr %irows_, align 4
  %add270 = add nsw i32 %199, %200
  %201 = load i32, ptr %dilated_in_dim, align 4
  %sub271 = sub nsw i32 %add270, %201
  br label %cond.end273

cond.false272:                                    ; preds = %cond.end264
  br label %cond.end273

cond.end273:                                      ; preds = %cond.false272, %cond.true269
  %cond274 = phi i32 [ %sub271, %cond.true269 ], [ 0, %cond.false272 ]
  store i32 %cond274, ptr %dpad, align 4
  %202 = load i32, ptr %input_dilated, align 4
  %tobool275 = icmp ne i32 %202, 0
  br i1 %tobool275, label %if.then276, label %if.end314

if.then276:                                       ; preds = %cond.end273
  %203 = load i32, ptr %lpad, align 4
  %cmp277 = icmp eq i32 %203, 0
  br i1 %cmp277, label %land.rhs279, label %land.end283

land.rhs279:                                      ; preds = %if.then276
  %204 = load i32, ptr %icol, align 4
  %rem280 = srem i32 %204, 2
  %cmp281 = icmp ne i32 %rem280, 0
  br label %land.end283

land.end283:                                      ; preds = %land.rhs279, %if.then276
  %205 = phi i1 [ false, %if.then276 ], [ %cmp281, %land.rhs279 ]
  %land.ext = zext i1 %205 to i32
  %206 = load i32, ptr %lpad, align 4
  %add284 = add nsw i32 %206, %land.ext
  store i32 %add284, ptr %lpad, align 4
  %207 = load i32, ptr %rpad, align 4
  %cmp285 = icmp eq i32 %207, 0
  br i1 %cmp285, label %land.rhs287, label %land.end292

land.rhs287:                                      ; preds = %land.end283
  %208 = load i32, ptr %icol, align 4
  %209 = load i32, ptr %icols_, align 4
  %add288 = add nsw i32 %208, %209
  %rem289 = srem i32 %add288, 2
  %cmp290 = icmp ne i32 %rem289, 1
  br label %land.end292

land.end292:                                      ; preds = %land.rhs287, %land.end283
  %210 = phi i1 [ false, %land.end283 ], [ %cmp290, %land.rhs287 ]
  %land.ext293 = zext i1 %210 to i32
  %211 = load i32, ptr %rpad, align 4
  %add294 = add nsw i32 %211, %land.ext293
  store i32 %add294, ptr %rpad, align 4
  %212 = load i32, ptr %upad, align 4
  %cmp295 = icmp eq i32 %212, 0
  br i1 %cmp295, label %land.rhs297, label %land.end301

land.rhs297:                                      ; preds = %land.end292
  %213 = load i32, ptr %irow, align 4
  %rem298 = srem i32 %213, 2
  %cmp299 = icmp ne i32 %rem298, 0
  br label %land.end301

land.end301:                                      ; preds = %land.rhs297, %land.end292
  %214 = phi i1 [ false, %land.end292 ], [ %cmp299, %land.rhs297 ]
  %land.ext302 = zext i1 %214 to i32
  %215 = load i32, ptr %upad, align 4
  %add303 = add nsw i32 %215, %land.ext302
  store i32 %add303, ptr %upad, align 4
  %216 = load i32, ptr %dpad, align 4
  %cmp304 = icmp eq i32 %216, 0
  br i1 %cmp304, label %land.rhs306, label %land.end311

land.rhs306:                                      ; preds = %land.end301
  %217 = load i32, ptr %irow, align 4
  %218 = load i32, ptr %irows_, align 4
  %add307 = add nsw i32 %217, %218
  %rem308 = srem i32 %add307, 2
  %cmp309 = icmp ne i32 %rem308, 1
  br label %land.end311

land.end311:                                      ; preds = %land.rhs306, %land.end301
  %219 = phi i1 [ false, %land.end301 ], [ %cmp309, %land.rhs306 ]
  %land.ext312 = zext i1 %219 to i32
  %220 = load i32, ptr %dpad, align 4
  %add313 = add nsw i32 %220, %land.ext312
  store i32 %add313, ptr %dpad, align 4
  br label %if.end314

if.end314:                                        ; preds = %land.end311, %cond.end273
  %221 = load i32, ptr %krow, align 4
  store i32 %221, ptr %krow_, align 4
  %222 = load i32, ptr %kcol, align 4
  store i32 %222, ptr %kcol_, align 4
  %223 = load i8, ptr %wrot180.addr, align 1
  %tobool315 = trunc i8 %223 to i1
  br i1 %tobool315, label %if.then316, label %if.end321

if.then316:                                       ; preds = %if.end314
  %224 = load i32, ptr %kernel_dim.addr, align 4
  %225 = load i32, ptr %krow, align 4
  %sub317 = sub nsw i32 %224, %225
  %226 = load i32, ptr %krows_, align 4
  %sub318 = sub nsw i32 %sub317, %226
  store i32 %sub318, ptr %krow_, align 4
  %227 = load i32, ptr %kernel_dim.addr, align 4
  %228 = load i32, ptr %kcol, align 4
  %sub319 = sub nsw i32 %227, %228
  %229 = load i32, ptr %kcols_, align 4
  %sub320 = sub nsw i32 %sub319, %229
  store i32 %sub320, ptr %kcol_, align 4
  br label %if.end321

if.end321:                                        ; preds = %if.then316, %if.end314
  %230 = load ptr, ptr %weights.addr, align 8
  %231 = load i32, ptr %krow_, align 4
  %232 = load i32, ptr %kernel_dim.addr, align 4
  %mul322 = mul nsw i32 %231, %232
  %233 = load i32, ptr %in_channels.addr, align 4
  %mul323 = mul nsw i32 %mul322, %233
  %234 = load i32, ptr %kcol_, align 4
  %235 = load i32, ptr %in_channels.addr, align 4
  %mul324 = mul nsw i32 %234, %235
  %add325 = add nsw i32 %mul323, %mul324
  %236 = load i32, ptr %kch, align 4
  %add326 = add nsw i32 %add325, %236
  %237 = load i32, ptr %out_channels.addr, align 4
  %mul327 = mul nsw i32 %add326, %237
  %idx.ext328 = sext i32 %mul327 to i64
  %add.ptr329 = getelementptr inbounds i8, ptr %230, i64 %idx.ext328
  %238 = load i32, ptr %poch, align 4
  %idx.ext330 = sext i32 %238 to i64
  %add.ptr331 = getelementptr inbounds i8, ptr %add.ptr329, i64 %idx.ext330
  store ptr %add.ptr331, ptr %weights_slice, align 8
  %239 = load i8, ptr %trans_weight_1203.addr, align 1
  %tobool332 = trunc i8 %239 to i1
  br i1 %tobool332, label %if.then333, label %if.else

if.then333:                                       ; preds = %if.end321
  %240 = load ptr, ptr %weights.addr, align 8
  %241 = load i32, ptr %kch, align 4
  %242 = load i32, ptr %kernel_dim.addr, align 4
  %mul334 = mul nsw i32 %241, %242
  %243 = load i32, ptr %kernel_dim.addr, align 4
  %mul335 = mul nsw i32 %mul334, %243
  %244 = load i32, ptr %krow_, align 4
  %245 = load i32, ptr %kernel_dim.addr, align 4
  %mul336 = mul nsw i32 %244, %245
  %add337 = add nsw i32 %mul335, %mul336
  %246 = load i32, ptr %kcol_, align 4
  %add338 = add nsw i32 %add337, %246
  %247 = load i32, ptr %out_channels.addr, align 4
  %mul339 = mul nsw i32 %add338, %247
  %idx.ext340 = sext i32 %mul339 to i64
  %add.ptr341 = getelementptr inbounds i8, ptr %240, i64 %idx.ext340
  %248 = load i32, ptr %poch, align 4
  %idx.ext342 = sext i32 %248 to i64
  %add.ptr343 = getelementptr inbounds i8, ptr %add.ptr341, i64 %idx.ext342
  store ptr %add.ptr343, ptr %weights_slice, align 8
  br label %if.end357

if.else:                                          ; preds = %if.end321
  %249 = load i8, ptr %trans_weight_0132.addr, align 1
  %tobool344 = trunc i8 %249 to i1
  br i1 %tobool344, label %if.then345, label %if.end356

if.then345:                                       ; preds = %if.else
  %250 = load ptr, ptr %weights.addr, align 8
  %251 = load i32, ptr %krow_, align 4
  %252 = load i32, ptr %kernel_dim.addr, align 4
  %mul346 = mul nsw i32 %251, %252
  %253 = load i32, ptr %out_channels.addr, align 4
  %mul347 = mul nsw i32 %mul346, %253
  %254 = load i32, ptr %kcol_, align 4
  %255 = load i32, ptr %out_channels.addr, align 4
  %mul348 = mul nsw i32 %254, %255
  %add349 = add nsw i32 %mul347, %mul348
  %256 = load i32, ptr %poch, align 4
  %add350 = add nsw i32 %add349, %256
  %257 = load i32, ptr %in_channels.addr, align 4
  %mul351 = mul nsw i32 %add350, %257
  %idx.ext352 = sext i32 %mul351 to i64
  %add.ptr353 = getelementptr inbounds i8, ptr %250, i64 %idx.ext352
  %258 = load i32, ptr %kch, align 4
  %idx.ext354 = sext i32 %258 to i64
  %add.ptr355 = getelementptr inbounds i8, ptr %add.ptr353, i64 %idx.ext354
  store ptr %add.ptr355, ptr %weights_slice, align 8
  br label %if.end356

if.end356:                                        ; preds = %if.then345, %if.else
  br label %if.end357

if.end357:                                        ; preds = %if.end356, %if.then333
  %259 = load ptr, ptr %input.addr, align 8
  %260 = load i32, ptr %b, align 4
  %261 = load i32, ptr %in_dim.addr, align 4
  %mul358 = mul nsw i32 %260, %261
  %262 = load i32, ptr %in_dim.addr, align 4
  %mul359 = mul nsw i32 %mul358, %262
  %263 = load i32, ptr %irow, align 4
  %264 = load i32, ptr %upad, align 4
  %add360 = add nsw i32 %263, %264
  %265 = load i32, ptr %input_dilated, align 4
  %shr = ashr i32 %add360, %265
  %266 = load i32, ptr %in_dim.addr, align 4
  %mul361 = mul nsw i32 %shr, %266
  %add362 = add nsw i32 %mul359, %mul361
  %267 = load i32, ptr %icol, align 4
  %268 = load i32, ptr %lpad, align 4
  %add363 = add nsw i32 %267, %268
  %269 = load i32, ptr %input_dilated, align 4
  %shr364 = ashr i32 %add363, %269
  %add365 = add nsw i32 %add362, %shr364
  %270 = load i32, ptr %in_channels.addr, align 4
  %mul366 = mul nsw i32 %add365, %270
  %idx.ext367 = sext i32 %mul366 to i64
  %add.ptr368 = getelementptr inbounds i8, ptr %259, i64 %idx.ext367
  %271 = load i32, ptr %kch, align 4
  %idx.ext369 = sext i32 %271 to i64
  %add.ptr370 = getelementptr inbounds i8, ptr %add.ptr368, i64 %idx.ext369
  store ptr %add.ptr370, ptr %in, align 8
  %272 = load i8, ptr %trans_input_3120.addr, align 1
  %tobool371 = trunc i8 %272 to i1
  br i1 %tobool371, label %if.then372, label %if.end387

if.then372:                                       ; preds = %if.end357
  %273 = load ptr, ptr %input.addr, align 8
  %274 = load i32, ptr %kch, align 4
  %275 = load i32, ptr %in_dim.addr, align 4
  %mul373 = mul nsw i32 %274, %275
  %276 = load i32, ptr %in_dim.addr, align 4
  %mul374 = mul nsw i32 %mul373, %276
  %277 = load i32, ptr %irow, align 4
  %278 = load i32, ptr %upad, align 4
  %add375 = add nsw i32 %277, %278
  %279 = load i32, ptr %input_dilated, align 4
  %shr376 = ashr i32 %add375, %279
  %280 = load i32, ptr %in_dim.addr, align 4
  %mul377 = mul nsw i32 %shr376, %280
  %add378 = add nsw i32 %mul374, %mul377
  %281 = load i32, ptr %icol, align 4
  %282 = load i32, ptr %lpad, align 4
  %add379 = add nsw i32 %281, %282
  %283 = load i32, ptr %input_dilated, align 4
  %shr380 = ashr i32 %add379, %283
  %add381 = add nsw i32 %add378, %shr380
  %284 = load i32, ptr %batch_size.addr, align 4
  %mul382 = mul nsw i32 %add381, %284
  %idx.ext383 = sext i32 %mul382 to i64
  %add.ptr384 = getelementptr inbounds i8, ptr %273, i64 %idx.ext383
  %285 = load i32, ptr %b, align 4
  %idx.ext385 = sext i32 %285 to i64
  %add.ptr386 = getelementptr inbounds i8, ptr %add.ptr384, i64 %idx.ext385
  store ptr %add.ptr386, ptr %in, align 8
  br label %if.end387

if.end387:                                        ; preds = %if.then372, %if.end357
  %286 = load i32, ptr %batch_size.addr, align 4
  %287 = load i32, ptr %in_dim.addr, align 4
  %288 = load i32, ptr %in_channels.addr, align 4
  %289 = load i32, ptr %out_channels.addr, align 4
  %290 = load i32, ptr %out_dim.addr, align 4
  %291 = load i32, ptr %pool_out_dim, align 4
  %292 = load i32, ptr %stride.addr, align 4
  %293 = load i32, ptr %padding.addr, align 4
  %294 = load i32, ptr %kernel_dim.addr, align 4
  %295 = load i32, ptr %kernel_dilation.addr, align 4
  %296 = load i32, ptr %pool_size.addr, align 4
  %297 = load i32, ptr %pool_stride.addr, align 4
  %298 = load i32, ptr %pool_padding.addr, align 4
  %299 = load i32, ptr %batches_, align 4
  %300 = load i32, ptr %porows_, align 4
  %301 = load i32, ptr %pocols_, align 4
  %302 = load i32, ptr %pochs_, align 4
  %303 = load i32, ptr %krows_, align 4
  %304 = load i32, ptr %kcols_, align 4
  %305 = load i32, ptr %kchs_, align 4
  %306 = load i32, ptr %lpad, align 4
  %307 = load i32, ptr %rpad, align 4
  %308 = load i32, ptr %upad, align 4
  %309 = load i32, ptr %dpad, align 4
  %310 = load i32, ptr %plpad, align 4
  %311 = load i32, ptr %prpad, align 4
  %312 = load i32, ptr %pupad, align 4
  %313 = load i32, ptr %pdpad, align 4
  %314 = load ptr, ptr %in, align 8
  %315 = load ptr, ptr %weights_slice, align 8
  %316 = load ptr, ptr %out, align 8
  %317 = load ptr, ptr %bias_, align 8
  %318 = load i32, ptr %act.addr, align 4
  %319 = load float, ptr %scale.addr, align 4
  %320 = load i8, ptr %wrot180.addr, align 1
  %tobool388 = trunc i8 %320 to i1
  %321 = load i8, ptr %trans_output_1203.addr, align 1
  %tobool389 = trunc i8 %321 to i1
  %322 = load i8, ptr %trans_input_3120.addr, align 1
  %tobool390 = trunc i8 %322 to i1
  %323 = load i8, ptr %trans_weight_1203.addr, align 1
  %tobool391 = trunc i8 %323 to i1
  %324 = load i8, ptr %trans_weight_0132.addr, align 1
  %tobool392 = trunc i8 %324 to i1
  %325 = load i8, ptr %no_bias, align 1
  %tobool393 = trunc i8 %325 to i1
  %326 = load i8, ptr %no_pool, align 1
  %tobool394 = trunc i8 %326 to i1
  %327 = load i8, ptr %downsample, align 1
  %tobool395 = trunc i8 %327 to i1
  %328 = load i32, ptr %input_dilated, align 4
  %tobool396 = icmp ne i32 %328, 0
  call void @sp_tiled_conv(i32 noundef signext %286, i32 noundef signext %287, i32 noundef signext %288, i32 noundef signext %289, i32 noundef signext %290, i32 noundef signext %291, i32 noundef signext %292, i32 noundef signext %293, i32 noundef %294, i32 noundef %295, i32 noundef %296, i32 noundef %297, i32 noundef %298, i32 noundef %299, i32 noundef %300, i32 noundef %301, i32 noundef %302, i32 noundef %303, i32 noundef %304, i32 noundef %305, i32 noundef %306, i32 noundef %307, i32 noundef %308, i32 noundef %309, i32 noundef %310, i32 noundef %311, i32 noundef %312, i32 noundef %313, ptr noundef %314, ptr noundef %315, ptr noundef %316, ptr noundef %317, i32 noundef %318, float noundef %319, i1 noundef %tobool388, i1 noundef %tobool389, i1 noundef %tobool390, i1 noundef %tobool391, i1 noundef %tobool392, i1 noundef %tobool393, i1 noundef %tobool394, i1 noundef %tobool395, i1 noundef %tobool396, i1 noundef false)
  br label %for.inc

for.inc:                                          ; preds = %if.end387
  %329 = load i32, ptr %kchs.addr, align 4
  %330 = load i32, ptr %kch, align 4
  %add397 = add nsw i32 %330, %329
  store i32 %add397, ptr %kch, align 4
  br label %for.cond82, !llvm.loop !27

for.end:                                          ; preds = %for.cond82
  br label %for.inc398

for.inc398:                                       ; preds = %for.end
  %331 = load i32, ptr %kcols.addr, align 4
  %332 = load i32, ptr %kcol, align 4
  %add399 = add nsw i32 %332, %331
  store i32 %add399, ptr %kcol, align 4
  br label %for.cond68, !llvm.loop !28

for.end400:                                       ; preds = %for.cond68
  br label %for.inc401

for.inc401:                                       ; preds = %for.end400
  %333 = load i32, ptr %krows.addr, align 4
  %334 = load i32, ptr %krow, align 4
  %add402 = add nsw i32 %334, %333
  store i32 %add402, ptr %krow, align 4
  br label %for.cond54, !llvm.loop !29

for.end403:                                       ; preds = %for.cond54
  br label %for.inc404

for.inc404:                                       ; preds = %for.end403
  %335 = load i32, ptr %pochs.addr, align 4
  %336 = load i32, ptr %poch, align 4
  %add405 = add nsw i32 %336, %335
  store i32 %add405, ptr %poch, align 4
  br label %for.cond50, !llvm.loop !30

for.end406:                                       ; preds = %for.cond50
  br label %for.inc407

for.inc407:                                       ; preds = %for.end406
  %337 = load i32, ptr %pocols.addr, align 4
  %338 = load i32, ptr %pocol, align 4
  %add408 = add nsw i32 %338, %337
  store i32 %add408, ptr %pocol, align 4
  br label %for.cond44, !llvm.loop !31

for.end409:                                       ; preds = %for.cond44
  br label %for.inc410

for.inc410:                                       ; preds = %for.end409
  %339 = load i32, ptr %porows.addr, align 4
  %340 = load i32, ptr %porow, align 4
  %add411 = add nsw i32 %340, %339
  store i32 %add411, ptr %porow, align 4
  br label %for.cond38, !llvm.loop !32

for.end412:                                       ; preds = %for.cond38
  br label %for.inc413

for.inc413:                                       ; preds = %for.end412
  %341 = load i32, ptr %batches.addr, align 4
  %342 = load i32, ptr %b, align 4
  %add414 = add nsw i32 %342, %341
  store i32 %add414, ptr %b, align 4
  br label %for.cond, !llvm.loop !33

for.end415:                                       ; preds = %for.cond
  ret void
}

; Function Attrs: noinline nounwind optnone
define internal void @sp_tiled_conv(i32 noundef signext %batch_size, i32 noundef signext %in_dim, i32 noundef signext %in_channels, i32 noundef signext %out_channels, i32 noundef signext %out_dim, i32 noundef signext %pool_out_dim, i32 noundef signext %stride, i32 noundef signext %padding, i32 noundef %kernel_dim, i32 noundef %kernel_dilation, i32 noundef %pool_size, i32 noundef %pool_stride, i32 noundef %pool_padding, i32 noundef %batches, i32 noundef %porows, i32 noundef %pocols, i32 noundef %pochs, i32 noundef %krows, i32 noundef %kcols, i32 noundef %kchs, i32 noundef %lpad, i32 noundef %rpad, i32 noundef %upad, i32 noundef %dpad, i32 noundef %plpad, i32 noundef %prpad, i32 noundef %pupad, i32 noundef %pdpad, ptr noundef %input, ptr noundef %weights, ptr noundef %output, ptr noundef %bias, i32 noundef %act, float noundef %scale, i1 noundef %wrot180, i1 noundef %trans_output_1203, i1 noundef %trans_input_3120, i1 noundef %trans_weight_1203, i1 noundef %trans_weight_0132, i1 noundef %no_bias, i1 noundef %no_pool, i1 noundef %downsample, i1 noundef %input_dilated, i1 noundef %dw) #0 {
entry:
  %batch_size.addr = alloca i32, align 4
  %in_dim.addr = alloca i32, align 4
  %in_channels.addr = alloca i32, align 4
  %out_channels.addr = alloca i32, align 4
  %out_dim.addr = alloca i32, align 4
  %pool_out_dim.addr = alloca i32, align 4
  %stride.addr = alloca i32, align 4
  %padding.addr = alloca i32, align 4
  %kernel_dim.addr = alloca i32, align 4
  %kernel_dilation.addr = alloca i32, align 4
  %pool_size.addr = alloca i32, align 4
  %pool_stride.addr = alloca i32, align 4
  %pool_padding.addr = alloca i32, align 4
  %batches.addr = alloca i32, align 4
  %porows.addr = alloca i32, align 4
  %pocols.addr = alloca i32, align 4
  %pochs.addr = alloca i32, align 4
  %krows.addr = alloca i32, align 4
  %kcols.addr = alloca i32, align 4
  %kchs.addr = alloca i32, align 4
  %lpad.addr = alloca i32, align 4
  %rpad.addr = alloca i32, align 4
  %upad.addr = alloca i32, align 4
  %dpad.addr = alloca i32, align 4
  %plpad.addr = alloca i32, align 4
  %prpad.addr = alloca i32, align 4
  %pupad.addr = alloca i32, align 4
  %pdpad.addr = alloca i32, align 4
  %input.addr = alloca ptr, align 8
  %weights.addr = alloca ptr, align 8
  %output.addr = alloca ptr, align 8
  %bias.addr = alloca ptr, align 8
  %act.addr = alloca i32, align 4
  %scale.addr = alloca float, align 4
  %wrot180.addr = alloca i8, align 1
  %trans_output_1203.addr = alloca i8, align 1
  %trans_input_3120.addr = alloca i8, align 1
  %trans_weight_1203.addr = alloca i8, align 1
  %trans_weight_0132.addr = alloca i8, align 1
  %no_bias.addr = alloca i8, align 1
  %no_pool.addr = alloca i8, align 1
  %downsample.addr = alloca i8, align 1
  %input_dilated.addr = alloca i8, align 1
  %dw.addr = alloca i8, align 1
  %orows = alloca i32, align 4
  %ocols = alloca i32, align 4
  %ochs = alloca i32, align 4
  %dilated_krows = alloca i32, align 4
  %dilated_kcols = alloca i32, align 4
  %irows = alloca i32, align 4
  %icols = alloca i32, align 4
  %irows_unpadded = alloca i32, align 4
  %icols_unpadded = alloca i32, align 4
  %ichs = alloca i32, align 4
  %transposed = alloca i8, align 1
  %max_pixels_per_row = alloca i32, align 4
  %out_channels_per_bank = alloca i32, align 4
  %in_channels_per_bank = alloca i32, align 4
  %B_rows = alloca i32, align 4
  %A_sp_addr_start = alloca i32, align 4
  %B_sp_addr_start = alloca i32, align 4
  %D_sp_addr_start = alloca i32, align 4
  %C_sp_addr_start = alloca i32, align 4
  %inputAddr = alloca i64, align 8
  %outputAddr = alloca i64, align 8
  %weightsAddr = alloca i64, align 8
  %biasAddr = alloca i64, align 8
  store i32 %batch_size, ptr %batch_size.addr, align 4
  store i32 %in_dim, ptr %in_dim.addr, align 4
  store i32 %in_channels, ptr %in_channels.addr, align 4
  store i32 %out_channels, ptr %out_channels.addr, align 4
  store i32 %out_dim, ptr %out_dim.addr, align 4
  store i32 %pool_out_dim, ptr %pool_out_dim.addr, align 4
  store i32 %stride, ptr %stride.addr, align 4
  store i32 %padding, ptr %padding.addr, align 4
  store i32 %kernel_dim, ptr %kernel_dim.addr, align 4
  store i32 %kernel_dilation, ptr %kernel_dilation.addr, align 4
  store i32 %pool_size, ptr %pool_size.addr, align 4
  store i32 %pool_stride, ptr %pool_stride.addr, align 4
  store i32 %pool_padding, ptr %pool_padding.addr, align 4
  store i32 %batches, ptr %batches.addr, align 4
  store i32 %porows, ptr %porows.addr, align 4
  store i32 %pocols, ptr %pocols.addr, align 4
  store i32 %pochs, ptr %pochs.addr, align 4
  store i32 %krows, ptr %krows.addr, align 4
  store i32 %kcols, ptr %kcols.addr, align 4
  store i32 %kchs, ptr %kchs.addr, align 4
  store i32 %lpad, ptr %lpad.addr, align 4
  store i32 %rpad, ptr %rpad.addr, align 4
  store i32 %upad, ptr %upad.addr, align 4
  store i32 %dpad, ptr %dpad.addr, align 4
  store i32 %plpad, ptr %plpad.addr, align 4
  store i32 %prpad, ptr %prpad.addr, align 4
  store i32 %pupad, ptr %pupad.addr, align 4
  store i32 %pdpad, ptr %pdpad.addr, align 4
  store ptr %input, ptr %input.addr, align 8
  store ptr %weights, ptr %weights.addr, align 8
  store ptr %output, ptr %output.addr, align 8
  store ptr %bias, ptr %bias.addr, align 8
  store i32 %act, ptr %act.addr, align 4
  store float %scale, ptr %scale.addr, align 4
  %frombool = zext i1 %wrot180 to i8
  store i8 %frombool, ptr %wrot180.addr, align 1
  %frombool1 = zext i1 %trans_output_1203 to i8
  store i8 %frombool1, ptr %trans_output_1203.addr, align 1
  %frombool2 = zext i1 %trans_input_3120 to i8
  store i8 %frombool2, ptr %trans_input_3120.addr, align 1
  %frombool3 = zext i1 %trans_weight_1203 to i8
  store i8 %frombool3, ptr %trans_weight_1203.addr, align 1
  %frombool4 = zext i1 %trans_weight_0132 to i8
  store i8 %frombool4, ptr %trans_weight_0132.addr, align 1
  %frombool5 = zext i1 %no_bias to i8
  store i8 %frombool5, ptr %no_bias.addr, align 1
  %frombool6 = zext i1 %no_pool to i8
  store i8 %frombool6, ptr %no_pool.addr, align 1
  %frombool7 = zext i1 %downsample to i8
  store i8 %frombool7, ptr %downsample.addr, align 1
  %frombool8 = zext i1 %input_dilated to i8
  store i8 %frombool8, ptr %input_dilated.addr, align 1
  %frombool9 = zext i1 %dw to i8
  store i8 %frombool9, ptr %dw.addr, align 1
  %0 = load i8, ptr %dw.addr, align 1
  %tobool = trunc i8 %0 to i1
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 1, ptr %kchs.addr, align 4
  store i32 1, ptr %pochs.addr, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %1 = load i32, ptr %porows.addr, align 4
  %2 = load i32, ptr %pool_stride.addr, align 4
  %mul = mul nsw i32 %1, %2
  %3 = load i32, ptr %pool_size.addr, align 4
  %add = add nsw i32 %mul, %3
  %sub = sub nsw i32 %add, 1
  %4 = load i32, ptr %pupad.addr, align 4
  %sub10 = sub nsw i32 %sub, %4
  %5 = load i32, ptr %pdpad.addr, align 4
  %sub11 = sub nsw i32 %sub10, %5
  store i32 %sub11, ptr %orows, align 4
  %6 = load i32, ptr %pocols.addr, align 4
  %7 = load i32, ptr %pool_stride.addr, align 4
  %mul12 = mul nsw i32 %6, %7
  %8 = load i32, ptr %pool_size.addr, align 4
  %add13 = add nsw i32 %mul12, %8
  %sub14 = sub nsw i32 %add13, 1
  %9 = load i32, ptr %plpad.addr, align 4
  %sub15 = sub nsw i32 %sub14, %9
  %10 = load i32, ptr %prpad.addr, align 4
  %sub16 = sub nsw i32 %sub15, %10
  store i32 %sub16, ptr %ocols, align 4
  %11 = load i32, ptr %pochs.addr, align 4
  store i32 %11, ptr %ochs, align 4
  %12 = load i32, ptr %krows.addr, align 4
  %13 = load i32, ptr %kernel_dilation.addr, align 4
  %sub17 = sub nsw i32 %13, 1
  %14 = load i32, ptr %krows.addr, align 4
  %sub18 = sub nsw i32 %14, 1
  %mul19 = mul nsw i32 %sub17, %sub18
  %add20 = add nsw i32 %12, %mul19
  store i32 %add20, ptr %dilated_krows, align 4
  %15 = load i32, ptr %kcols.addr, align 4
  %16 = load i32, ptr %kernel_dilation.addr, align 4
  %sub21 = sub nsw i32 %16, 1
  %17 = load i32, ptr %kcols.addr, align 4
  %sub22 = sub nsw i32 %17, 1
  %mul23 = mul nsw i32 %sub21, %sub22
  %add24 = add nsw i32 %15, %mul23
  store i32 %add24, ptr %dilated_kcols, align 4
  %18 = load i32, ptr %orows, align 4
  %19 = load i32, ptr %stride.addr, align 4
  %mul25 = mul nsw i32 %18, %19
  %20 = load i32, ptr %dilated_krows, align 4
  %add26 = add nsw i32 %mul25, %20
  %sub27 = sub nsw i32 %add26, 1
  store i32 %sub27, ptr %irows, align 4
  %21 = load i32, ptr %ocols, align 4
  %22 = load i32, ptr %stride.addr, align 4
  %mul28 = mul nsw i32 %21, %22
  %23 = load i32, ptr %dilated_kcols, align 4
  %add29 = add nsw i32 %mul28, %23
  %sub30 = sub nsw i32 %add29, 1
  store i32 %sub30, ptr %icols, align 4
  %24 = load i32, ptr %irows, align 4
  %25 = load i32, ptr %upad.addr, align 4
  %sub31 = sub nsw i32 %24, %25
  %26 = load i32, ptr %dpad.addr, align 4
  %sub32 = sub nsw i32 %sub31, %26
  store i32 %sub32, ptr %irows_unpadded, align 4
  %27 = load i32, ptr %icols, align 4
  %28 = load i32, ptr %lpad.addr, align 4
  %sub33 = sub nsw i32 %27, %28
  %29 = load i32, ptr %rpad.addr, align 4
  %sub34 = sub nsw i32 %sub33, %29
  store i32 %sub34, ptr %icols_unpadded, align 4
  %30 = load i32, ptr %kchs.addr, align 4
  store i32 %30, ptr %ichs, align 4
  %31 = load i8, ptr %input_dilated.addr, align 1
  %tobool35 = trunc i8 %31 to i1
  br i1 %tobool35, label %if.then36, label %if.end68

if.then36:                                        ; preds = %if.end
  %32 = load i32, ptr %irows_unpadded, align 4
  %add37 = add nsw i32 %32, 1
  %div = sdiv i32 %add37, 2
  store i32 %div, ptr %irows_unpadded, align 4
  %33 = load i32, ptr %icols_unpadded, align 4
  %add38 = add nsw i32 %33, 1
  %div39 = sdiv i32 %add38, 2
  store i32 %div39, ptr %icols_unpadded, align 4
  %34 = load i32, ptr %irows_unpadded, align 4
  %35 = load i8, ptr %input_dilated.addr, align 1
  %tobool40 = trunc i8 %35 to i1
  br i1 %tobool40, label %cond.true, label %cond.false

cond.true:                                        ; preds = %if.then36
  %36 = load i32, ptr %upad.addr, align 4
  %add41 = add nsw i32 %36, 1
  %div42 = sdiv i32 %add41, 2
  br label %cond.end

cond.false:                                       ; preds = %if.then36
  %37 = load i32, ptr %upad.addr, align 4
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %div42, %cond.true ], [ %37, %cond.false ]
  %add43 = add nsw i32 %34, %cond
  %38 = load i8, ptr %input_dilated.addr, align 1
  %tobool44 = trunc i8 %38 to i1
  br i1 %tobool44, label %cond.true45, label %cond.false48

cond.true45:                                      ; preds = %cond.end
  %39 = load i32, ptr %dpad.addr, align 4
  %add46 = add nsw i32 %39, 1
  %div47 = sdiv i32 %add46, 2
  br label %cond.end49

cond.false48:                                     ; preds = %cond.end
  %40 = load i32, ptr %dpad.addr, align 4
  br label %cond.end49

cond.end49:                                       ; preds = %cond.false48, %cond.true45
  %cond50 = phi i32 [ %div47, %cond.true45 ], [ %40, %cond.false48 ]
  %add51 = add nsw i32 %add43, %cond50
  store i32 %add51, ptr %irows, align 4
  %41 = load i32, ptr %icols_unpadded, align 4
  %42 = load i8, ptr %input_dilated.addr, align 1
  %tobool52 = trunc i8 %42 to i1
  br i1 %tobool52, label %cond.true53, label %cond.false56

cond.true53:                                      ; preds = %cond.end49
  %43 = load i32, ptr %lpad.addr, align 4
  %add54 = add nsw i32 %43, 1
  %div55 = sdiv i32 %add54, 2
  br label %cond.end57

cond.false56:                                     ; preds = %cond.end49
  %44 = load i32, ptr %lpad.addr, align 4
  br label %cond.end57

cond.end57:                                       ; preds = %cond.false56, %cond.true53
  %cond58 = phi i32 [ %div55, %cond.true53 ], [ %44, %cond.false56 ]
  %add59 = add nsw i32 %41, %cond58
  %45 = load i8, ptr %input_dilated.addr, align 1
  %tobool60 = trunc i8 %45 to i1
  br i1 %tobool60, label %cond.true61, label %cond.false64

cond.true61:                                      ; preds = %cond.end57
  %46 = load i32, ptr %rpad.addr, align 4
  %add62 = add nsw i32 %46, 1
  %div63 = sdiv i32 %add62, 2
  br label %cond.end65

cond.false64:                                     ; preds = %cond.end57
  %47 = load i32, ptr %rpad.addr, align 4
  br label %cond.end65

cond.end65:                                       ; preds = %cond.false64, %cond.true61
  %cond66 = phi i32 [ %div63, %cond.true61 ], [ %47, %cond.false64 ]
  %add67 = add nsw i32 %add59, %cond66
  store i32 %add67, ptr %icols, align 4
  br label %if.end68

if.end68:                                         ; preds = %cond.end65, %if.end
  %48 = load i8, ptr %trans_output_1203.addr, align 1
  %tobool69 = trunc i8 %48 to i1
  br i1 %tobool69, label %lor.end, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %if.end68
  %49 = load i8, ptr %trans_input_3120.addr, align 1
  %tobool70 = trunc i8 %49 to i1
  br i1 %tobool70, label %lor.end, label %lor.lhs.false71

lor.lhs.false71:                                  ; preds = %lor.lhs.false
  %50 = load i8, ptr %trans_weight_1203.addr, align 1
  %tobool72 = trunc i8 %50 to i1
  br i1 %tobool72, label %lor.end, label %lor.rhs

lor.rhs:                                          ; preds = %lor.lhs.false71
  %51 = load i8, ptr %trans_weight_0132.addr, align 1
  %tobool73 = trunc i8 %51 to i1
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %lor.lhs.false71, %lor.lhs.false, %if.end68
  %52 = phi i1 [ true, %lor.lhs.false71 ], [ true, %lor.lhs.false ], [ true, %if.end68 ], [ %tobool73, %lor.rhs ]
  %frombool74 = zext i1 %52 to i8
  store i8 %frombool74, ptr %transposed, align 1
  %53 = load i8, ptr %transposed, align 1
  %tobool75 = trunc i8 %53 to i1
  br i1 %tobool75, label %cond.true85, label %lor.lhs.false76

lor.lhs.false76:                                  ; preds = %lor.end
  %54 = load i8, ptr %wrot180.addr, align 1
  %tobool77 = trunc i8 %54 to i1
  br i1 %tobool77, label %cond.true85, label %lor.lhs.false78

lor.lhs.false78:                                  ; preds = %lor.lhs.false76
  %55 = load i8, ptr %downsample.addr, align 1
  %tobool79 = trunc i8 %55 to i1
  br i1 %tobool79, label %cond.true85, label %lor.lhs.false80

lor.lhs.false80:                                  ; preds = %lor.lhs.false78
  %56 = load i8, ptr %input_dilated.addr, align 1
  %tobool81 = trunc i8 %56 to i1
  br i1 %tobool81, label %cond.true85, label %lor.lhs.false82

lor.lhs.false82:                                  ; preds = %lor.lhs.false80
  %57 = load i32, ptr %kernel_dilation.addr, align 4
  %cmp = icmp sgt i32 %57, 1
  br i1 %cmp, label %cond.true85, label %lor.lhs.false83

lor.lhs.false83:                                  ; preds = %lor.lhs.false82
  %58 = load i32, ptr %ichs, align 4
  %cmp84 = icmp sgt i32 %58, 16
  br i1 %cmp84, label %cond.true85, label %cond.false86

cond.true85:                                      ; preds = %lor.lhs.false83, %lor.lhs.false82, %lor.lhs.false80, %lor.lhs.false78, %lor.lhs.false76, %lor.end
  br label %cond.end88

cond.false86:                                     ; preds = %lor.lhs.false83
  %59 = load i32, ptr %ichs, align 4
  %div87 = sdiv i32 16, %59
  br label %cond.end88

cond.end88:                                       ; preds = %cond.false86, %cond.true85
  %cond89 = phi i32 [ 1, %cond.true85 ], [ %div87, %cond.false86 ]
  store i32 %cond89, ptr %max_pixels_per_row, align 4
  %60 = load i32, ptr %max_pixels_per_row, align 4
  %61 = load i32, ptr %kcols.addr, align 4
  %cmp90 = icmp sgt i32 %60, %61
  br i1 %cmp90, label %if.then91, label %if.end92

if.then91:                                        ; preds = %cond.end88
  %62 = load i32, ptr %kcols.addr, align 4
  store i32 %62, ptr %max_pixels_per_row, align 4
  br label %if.end92

if.end92:                                         ; preds = %if.then91, %cond.end88
  %63 = load i32, ptr %ochs, align 4
  %div93 = sdiv i32 %63, 16
  %64 = load i32, ptr %ochs, align 4
  %rem = srem i32 %64, 16
  %cmp94 = icmp ne i32 %rem, 0
  %conv = zext i1 %cmp94 to i32
  %add95 = add nsw i32 %div93, %conv
  store i32 %add95, ptr %out_channels_per_bank, align 4
  %65 = load i32, ptr %kchs.addr, align 4
  %div96 = sdiv i32 %65, 16
  %66 = load i32, ptr %kchs.addr, align 4
  %rem97 = srem i32 %66, 16
  %cmp98 = icmp ne i32 %rem97, 0
  %conv99 = zext i1 %cmp98 to i32
  %add100 = add nsw i32 %div96, %conv99
  store i32 %add100, ptr %in_channels_per_bank, align 4
  %67 = load i8, ptr %trans_weight_0132.addr, align 1
  %tobool101 = trunc i8 %67 to i1
  br i1 %tobool101, label %cond.true103, label %cond.false107

cond.true103:                                     ; preds = %if.end92
  %68 = load i32, ptr %in_channels_per_bank, align 4
  %69 = load i32, ptr %kcols.addr, align 4
  %mul104 = mul nsw i32 %68, %69
  %70 = load i32, ptr %krows.addr, align 4
  %mul105 = mul nsw i32 %mul104, %70
  %71 = load i32, ptr %ochs, align 4
  %mul106 = mul nsw i32 %mul105, %71
  br label %cond.end111

cond.false107:                                    ; preds = %if.end92
  %72 = load i32, ptr %out_channels_per_bank, align 4
  %73 = load i32, ptr %kcols.addr, align 4
  %mul108 = mul nsw i32 %72, %73
  %74 = load i32, ptr %krows.addr, align 4
  %mul109 = mul nsw i32 %mul108, %74
  %75 = load i32, ptr %kchs.addr, align 4
  %mul110 = mul nsw i32 %mul109, %75
  br label %cond.end111

cond.end111:                                      ; preds = %cond.false107, %cond.true103
  %cond112 = phi i32 [ %mul106, %cond.true103 ], [ %mul110, %cond.false107 ]
  store i32 %cond112, ptr %B_rows, align 4
  store i32 0, ptr %A_sp_addr_start, align 4
  %76 = load i32, ptr %B_rows, align 4
  %sub113 = sub nsw i32 16384, %76
  store i32 %sub113, ptr %B_sp_addr_start, align 4
  %77 = load i32, ptr @sp_tiled_conv.D_sp_addr_row, align 4
  %add114 = add i32 -2147483648, %77
  store i32 %add114, ptr %D_sp_addr_start, align 4
  %78 = load i32, ptr @sp_tiled_conv.C_sp_addr_row, align 4
  %add115 = add i32 -1073741824, %78
  store i32 %add115, ptr %C_sp_addr_start, align 4
  %79 = load ptr, ptr %bias.addr, align 8
  %cmp116 = icmp ne ptr %79, null
  br i1 %cmp116, label %if.then118, label %if.end121

if.then118:                                       ; preds = %cond.end111
  %80 = load i32, ptr @sp_tiled_conv.D_sp_addr_row, align 4
  %add119 = add i32 %80, 512
  %rem120 = urem i32 %add119, 1024
  store i32 %rem120, ptr @sp_tiled_conv.D_sp_addr_row, align 4
  br label %if.end121

if.end121:                                        ; preds = %if.then118, %cond.end111
  %81 = load ptr, ptr %output.addr, align 8
  %cmp122 = icmp ne ptr %81, null
  br i1 %cmp122, label %if.then124, label %if.end127

if.then124:                                       ; preds = %if.end121
  %82 = load i32, ptr @sp_tiled_conv.C_sp_addr_row, align 4
  %add125 = add i32 %82, 512
  %rem126 = urem i32 %add125, 1024
  store i32 %rem126, ptr @sp_tiled_conv.C_sp_addr_row, align 4
  br label %if.end127

if.end127:                                        ; preds = %if.then124, %if.end121
  %83 = load ptr, ptr %input.addr, align 8
  %84 = ptrtoint ptr %83 to i64
  store i64 %84, ptr %inputAddr, align 8
  %85 = load ptr, ptr %output.addr, align 8
  %86 = ptrtoint ptr %85 to i64
  store i64 %86, ptr %outputAddr, align 8
  %87 = load ptr, ptr %weights.addr, align 8
  %88 = ptrtoint ptr %87 to i64
  store i64 %88, ptr %weightsAddr, align 8
  %89 = load ptr, ptr %bias.addr, align 8
  %90 = ptrtoint ptr %89 to i64
  store i64 %90, ptr %biasAddr, align 8
  call void @llvm.riscv.loopConvWsConfig1(i64 5348101868027906, i64 281483567235081)
  %call = call signext i32 (ptr, ...) @printf(ptr noundef @.str.18)
  call void @llvm.riscv.loopConvWsConfig2(i64 844429225164800, i64 562988608716819)
  %call128 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.19)
  call void @llvm.riscv.loopConvWsConfig3(i64 844437816213505, i64 562954248519680)
  %call129 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.20)
  call void @llvm.riscv.loopConvWsConfig4(i64 2533274790395904, i64 65545)
  %call130 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.21)
  call void @llvm.riscv.loopConvWsConfig5(i64 %88, i64 %86)
  %call131 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.22)
  call void @llvm.riscv.loopConvWsConfig6(i64 %90, i64 %84)
  %call132 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.23)
  call void @llvm.riscv.loopConvWs(i64 256, i64 1)
  %call133 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.24)
  ret void
}

attributes #0 = { noinline nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+m,+relax,-save-restore" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+m,+relax,-save-restore" }
attributes #2 = { noreturn "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+m,+relax,-save-restore" }
attributes #3 = { noreturn }
attributes #4 = { nounwind }

declare void @llvm.riscv.configSt(i64, i64)
declare void @llvm.riscv.configEx(i64, i64)
declare void @llvm.riscv.loopConvWsConfig1(i64, i64)
declare void @llvm.riscv.loopConvWsConfig2(i64, i64)
declare void @llvm.riscv.loopConvWsConfig3(i64, i64)
declare void @llvm.riscv.loopConvWsConfig4(i64, i64)
declare void @llvm.riscv.loopConvWsConfig5(i64, i64)
declare void @llvm.riscv.loopConvWsConfig6(i64, i64)
declare void @llvm.riscv.loopConvWs(i64, i64)

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
!21 = distinct !{!21, !6}
!22 = distinct !{!22, !6}
!23 = distinct !{!23, !6}
!24 = distinct !{!24, !6}
!25 = distinct !{!25, !6}
!26 = !{i64 22599}
!27 = distinct !{!27, !6}
!28 = distinct !{!28, !6}
!29 = distinct !{!29, !6}
!30 = distinct !{!30, !6}
!31 = distinct !{!31, !6}
!32 = distinct !{!32, !6}
!33 = distinct !{!33, !6}
