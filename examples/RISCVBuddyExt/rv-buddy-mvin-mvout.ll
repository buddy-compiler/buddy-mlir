@.str = private unnamed_addr constant [15 x i8] c"print arrayB.\0A\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"%d \00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@.str.3 = private unnamed_addr constant [8 x i8] c"flush.\0A\00", align 1
@.str.4 = private unnamed_addr constant [12 x i8] c"config_ld.\0A\00", align 1
@.str.5 = private unnamed_addr constant [12 x i8] c"config_st.\0A\00", align 1
@.str.6 = private unnamed_addr constant [7 x i8] c"mvin.\0A\00", align 1
@.str.7 = private unnamed_addr constant [8 x i8] c"mvout.\0A\00", align 1

define dso_local signext i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %arrayA = alloca [16 x [16 x i8]], align 16
  %n = alloca i32, align 4
  %m = alloca i32, align 4
  %arrayB = alloca [16 x [16 x i8]], align 16
  %n9 = alloca i32, align 4
  %m13 = alloca i32, align 4
  %addrA = alloca i64, align 8
  %addrB = alloca i64, align 8
  %n36 = alloca i32, align 4
  %m41 = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  store i32 0, ptr %n, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc6, %entry
  %0 = load i32, ptr %n, align 4
  %cmp = icmp slt i32 %0, 16
  br i1 %cmp, label %for.body, label %for.end8

for.body:                                         ; preds = %for.cond
  store i32 0, ptr %m, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %1 = load i32, ptr %m, align 4
  %cmp2 = icmp slt i32 %1, 16
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %2 = load i32, ptr %n, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds [16 x [16 x i8]], ptr %arrayA, i64 0, i64 %idxprom
  %3 = load i32, ptr %m, align 4
  %idxprom4 = sext i32 %3 to i64
  %arrayidx5 = getelementptr inbounds [16 x i8], ptr %arrayidx, i64 0, i64 %idxprom4
  store i8 1, ptr %arrayidx5, align 1
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %4 = load i32, ptr %m, align 4
  %inc = add nsw i32 %4, 1
  store i32 %inc, ptr %m, align 4
  br label %for.cond1, !llvm.loop !5

for.end:                                          ; preds = %for.cond1
  br label %for.inc6

for.inc6:                                         ; preds = %for.end
  %5 = load i32, ptr %n, align 4
  %inc7 = add nsw i32 %5, 1
  store i32 %inc7, ptr %n, align 4
  br label %for.cond, !llvm.loop !7

for.end8:                                         ; preds = %for.cond
  %call = call signext i32 (ptr, ...) @printf(ptr noundef @.str)
  store i32 0, ptr %n9, align 4
  br label %for.cond10

for.cond10:                                       ; preds = %for.inc26, %for.end8
  %6 = load i32, ptr %n9, align 4
  %cmp11 = icmp slt i32 %6, 16
  br i1 %cmp11, label %for.body12, label %for.end28

for.body12:                                       ; preds = %for.cond10
  store i32 0, ptr %m13, align 4
  br label %for.cond14

for.cond14:                                       ; preds = %for.inc22, %for.body12
  %7 = load i32, ptr %m13, align 4
  %cmp15 = icmp slt i32 %7, 16
  br i1 %cmp15, label %for.body16, label %for.end24

for.body16:                                       ; preds = %for.cond14
  %8 = load i32, ptr %n9, align 4
  %idxprom17 = sext i32 %8 to i64
  %arrayidx18 = getelementptr inbounds [16 x [16 x i8]], ptr %arrayB, i64 0, i64 %idxprom17
  %9 = load i32, ptr %m13, align 4
  %idxprom19 = sext i32 %9 to i64
  %arrayidx20 = getelementptr inbounds [16 x i8], ptr %arrayidx18, i64 0, i64 %idxprom19
  %10 = load i8, ptr %arrayidx20, align 1
  %conv = sext i8 %10 to i32
  %call21 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef signext %conv)
  br label %for.inc22

for.inc22:                                        ; preds = %for.body16
  %11 = load i32, ptr %m13, align 4
  %inc23 = add nsw i32 %11, 1
  store i32 %inc23, ptr %m13, align 4
  br label %for.cond14, !llvm.loop !8

for.end24:                                        ; preds = %for.cond14
  %call25 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.2)
  br label %for.inc26

for.inc26:                                        ; preds = %for.end24
  %12 = load i32, ptr %n9, align 4
  %inc27 = add nsw i32 %12, 1
  store i32 %inc27, ptr %n9, align 4
  br label %for.cond10, !llvm.loop !9

for.end28:                                        ; preds = %for.cond10
  call void @llvm.riscv.flush(i64 0, i64 0)
  %call29 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.3)
  call void @llvm.riscv.configLd(i64 4575657221409472769, i64 16)
  %call30 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.4)
  call void @llvm.riscv.configSt(i64 2, i64 4575657221408423952)
  %call31 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.5)
  %arraydecay = getelementptr inbounds [16 x [16 x i8]], ptr %arrayA, i64 0, i64 0
  %13 = ptrtoint ptr %arraydecay to i64
  store i64 %13, ptr %addrA, align 8
  call void @llvm.riscv.mvin(i64 %13, i64 4503668346847232)
  %call32 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.6)
  %arraydecay33 = getelementptr inbounds [16 x [16 x i8]], ptr %arrayB, i64 0, i64 0
  %14 = ptrtoint ptr %arraydecay33 to i64
  store i64 %14, ptr %addrB, align 8
  call void @llvm.riscv.mvout(i64 %14, i64 4503668346847232)
  %call34 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.7)
  %call35 = call signext i32 (ptr, ...) @printf(ptr noundef @.str)
  store i32 0, ptr %n36, align 4
  br label %for.cond37

for.cond37:                                       ; preds = %for.inc56, %for.end28
  %15 = load i32, ptr %n36, align 4
  %cmp38 = icmp slt i32 %15, 16
  br i1 %cmp38, label %for.body40, label %for.end58

for.body40:                                       ; preds = %for.cond37
  store i32 0, ptr %m41, align 4
  br label %for.cond42

for.cond42:                                       ; preds = %for.inc52, %for.body40
  %16 = load i32, ptr %m41, align 4
  %cmp43 = icmp slt i32 %16, 16
  br i1 %cmp43, label %for.body45, label %for.end54

for.body45:                                       ; preds = %for.cond42
  %17 = load i32, ptr %n36, align 4
  %idxprom46 = sext i32 %17 to i64
  %arrayidx47 = getelementptr inbounds [16 x [16 x i8]], ptr %arrayB, i64 0, i64 %idxprom46
  %18 = load i32, ptr %m41, align 4
  %idxprom48 = sext i32 %18 to i64
  %arrayidx49 = getelementptr inbounds [16 x i8], ptr %arrayidx47, i64 0, i64 %idxprom48
  %19 = load i8, ptr %arrayidx49, align 1
  %conv50 = sext i8 %19 to i32
  %call51 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef signext %conv50)
  br label %for.inc52

for.inc52:                                        ; preds = %for.body45
  %20 = load i32, ptr %m41, align 4
  %inc53 = add nsw i32 %20, 1
  store i32 %inc53, ptr %m41, align 4
  br label %for.cond42, !llvm.loop !10

for.end54:                                        ; preds = %for.cond42
  %call55 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.2)
  br label %for.inc56

for.inc56:                                        ; preds = %for.end54
  %21 = load i32, ptr %n36, align 4
  %inc57 = add nsw i32 %21, 1
  store i32 %inc57, ptr %n36, align 4
  br label %for.cond37, !llvm.loop !11

for.end58:                                        ; preds = %for.cond37
  ret i32 0
}

declare dso_local signext i32 @printf(ptr noundef, ...) #1
declare void @llvm.riscv.mvin(i64, i64)
declare void @llvm.riscv.mvout(i64 ,i64)
declare void @llvm.riscv.flush(i64, i64)
declare void @llvm.riscv.configLd(i64, i64)
declare void @llvm.riscv.configSt(i64, i64)

attributes #0 = { noinline nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+m,+relax,-save-restore" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+m,+relax,-save-restore" }

!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.mustprogress"}
!7 = distinct !{!7, !6}
!8 = distinct !{!8, !6}
!9 = distinct !{!9, !6}
!10 = distinct !{!10, !6}
!11 = distinct !{!11, !6}
