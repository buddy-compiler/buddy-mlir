@.str = private unnamed_addr constant [12 x i8] c"config_ld.\0A\00", align 1
@.str.1 = private unnamed_addr constant [12 x i8] c"config_st.\0A\00", align 1
@.str.2 = private unnamed_addr constant [7 x i8] c"mvin.\0A\00", align 1
@.str.3 = private unnamed_addr constant [8 x i8] c"mvout.\0A\00", align 1
@.str.4 = private unnamed_addr constant [20 x i8] c"print input array.\0A\00", align 1
@.str.5 = private unnamed_addr constant [4 x i8] c"%d \00", align 1
@.str.6 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@.str.7 = private unnamed_addr constant [21 x i8] c"print output array.\0A\00", align 1

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %input = alloca [16 x [16 x i8]], align 1
  %output = alloca [13 x [14 x i8]], align 1
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %intputAddr = alloca i64, align 8
  %outputAddr = alloca i64, align 8
  %i14 = alloca i32, align 4
  %j18 = alloca i32, align 4
  %i35 = alloca i32, align 4
  %j40 = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc6, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 16
  br i1 %cmp, label %for.body, label %for.end8

for.body:                                         ; preds = %for.cond
  store i32 0, ptr %j, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %1 = load i32, ptr %j, align 4
  %cmp2 = icmp slt i32 %1, 16
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %2 = load i32, ptr %i, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds [16 x [16 x i8]], ptr %input, i64 0, i64 %idxprom
  %3 = load i32, ptr %j, align 4
  %idxprom4 = sext i32 %3 to i64
  %arrayidx5 = getelementptr inbounds [16 x i8], ptr %arrayidx, i64 0, i64 %idxprom4
  store i8 1, ptr %arrayidx5, align 1
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %4 = load i32, ptr %j, align 4
  %inc = add nsw i32 %4, 1
  store i32 %inc, ptr %j, align 4
  br label %for.cond1, !llvm.loop !5

for.end:                                          ; preds = %for.cond1
  br label %for.inc6

for.inc6:                                         ; preds = %for.end
  %5 = load i32, ptr %i, align 4
  %inc7 = add nsw i32 %5, 1
  store i32 %inc7, ptr %i, align 4
  br label %for.cond, !llvm.loop !7

for.end8:                                         ; preds = %for.cond
  %arraydecay = getelementptr inbounds [16 x [16 x i8]], ptr %input, i64 0, i64 0
  %6 = ptrtoint ptr %arraydecay to i64
  store i64 %6, ptr %intputAddr, align 8
  %arraydecay9 = getelementptr inbounds [13 x [14 x i8]], ptr %output, i64 0, i64 0
  %7 = ptrtoint ptr %arraydecay9 to i64
  store i64 %7, ptr %outputAddr, align 8
  call void @llvm.riscv.configLd(i64 4575657221409472769,i64 16)
  %call = call signext i32 (ptr, ...) @printf(ptr noundef @.str)
  call void @llvm.riscv.configSt(i64 2, i64 4575657221408423950)
  %call10 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.1)
  call void @llvm.riscv.mvin(i64 %6, i64 4503668346847232)
  %call11 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.2)
  call void @llvm.riscv.mvout(i64 %7, i64 3659234826780672)
  %call12 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.3)
  %call13 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.4)
  store i32 0, ptr %i14, align 4
  br label %for.cond15

for.cond15:                                       ; preds = %for.inc31, %for.end8
  %8 = load i32, ptr %i14, align 4
  %cmp16 = icmp slt i32 %8, 16
  br i1 %cmp16, label %for.body17, label %for.end33

for.body17:                                       ; preds = %for.cond15
  store i32 0, ptr %j18, align 4
  br label %for.cond19

for.cond19:                                       ; preds = %for.inc27, %for.body17
  %9 = load i32, ptr %j18, align 4
  %cmp20 = icmp slt i32 %9, 16
  br i1 %cmp20, label %for.body21, label %for.end29

for.body21:                                       ; preds = %for.cond19
  %10 = load i32, ptr %i14, align 4
  %idxprom22 = sext i32 %10 to i64
  %arrayidx23 = getelementptr inbounds [16 x [16 x i8]], ptr %input, i64 0, i64 %idxprom22
  %11 = load i32, ptr %j18, align 4
  %idxprom24 = sext i32 %11 to i64
  %arrayidx25 = getelementptr inbounds [16 x i8], ptr %arrayidx23, i64 0, i64 %idxprom24
  %12 = load i8, ptr %arrayidx25, align 1
  %conv = sext i8 %12 to i32
  %call26 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.5, i32 noundef signext %conv)
  br label %for.inc27

for.inc27:                                        ; preds = %for.body21
  %13 = load i32, ptr %j18, align 4
  %inc28 = add nsw i32 %13, 1
  store i32 %inc28, ptr %j18, align 4
  br label %for.cond19, !llvm.loop !8

for.end29:                                        ; preds = %for.cond19
  %call30 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.6)
  br label %for.inc31

for.inc31:                                        ; preds = %for.end29
  %14 = load i32, ptr %i14, align 4
  %inc32 = add nsw i32 %14, 1
  store i32 %inc32, ptr %i14, align 4
  br label %for.cond15, !llvm.loop !9

for.end33:                                        ; preds = %for.cond15
  %call34 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.7)
  store i32 0, ptr %i35, align 4
  br label %for.cond36

for.cond36:                                       ; preds = %for.inc55, %for.end33
  %15 = load i32, ptr %i35, align 4
  %cmp37 = icmp slt i32 %15, 13
  br i1 %cmp37, label %for.body39, label %for.end57

for.body39:                                       ; preds = %for.cond36
  store i32 0, ptr %j40, align 4
  br label %for.cond41

for.cond41:                                       ; preds = %for.inc51, %for.body39
  %16 = load i32, ptr %j40, align 4
  %cmp42 = icmp slt i32 %16, 14
  br i1 %cmp42, label %for.body44, label %for.end53

for.body44:                                       ; preds = %for.cond41
  %17 = load i32, ptr %i35, align 4
  %idxprom45 = sext i32 %17 to i64
  %arrayidx46 = getelementptr inbounds [13 x [14 x i8]], ptr %output, i64 0, i64 %idxprom45
  %18 = load i32, ptr %j40, align 4
  %idxprom47 = sext i32 %18 to i64
  %arrayidx48 = getelementptr inbounds [14 x i8], ptr %arrayidx46, i64 0, i64 %idxprom47
  %19 = load i8, ptr %arrayidx48, align 1
  %conv49 = sext i8 %19 to i32
  %call50 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.5, i32 noundef signext %conv49)
  br label %for.inc51

for.inc51:                                        ; preds = %for.body44
  %20 = load i32, ptr %j40, align 4
  %inc52 = add nsw i32 %20, 1
  store i32 %inc52, ptr %j40, align 4
  br label %for.cond41, !llvm.loop !10

for.end53:                                        ; preds = %for.cond41
  %call54 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.6)
  br label %for.inc55

for.inc55:                                        ; preds = %for.end53
  %21 = load i32, ptr %i35, align 4
  %inc56 = add nsw i32 %21, 1
  store i32 %inc56, ptr %i35, align 4
  br label %for.cond36, !llvm.loop !11

for.end57:                                        ; preds = %for.cond36
  ret i32 0
}

declare dso_local signext i32 @printf(ptr noundef, ...) #1

attributes #0 = { noinline nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+m,+relax,-save-restore" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+m,+relax,-save-restore" }
declare void @llvm.riscv.mvin(i64, i64)
declare void @llvm.riscv.mvout(i64, i64)
declare void @llvm.riscv.flush(i64, i64)
declare void @llvm.riscv.configSt(i64, i64)
declare void @llvm.riscv.configLd(i64, i64)

!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.mustprogress"}
!7 = distinct !{!7, !6}
!8 = distinct !{!8, !6}
!9 = distinct !{!9, !6}
!10 = distinct !{!10, !6}
!11 = distinct !{!11, !6}
