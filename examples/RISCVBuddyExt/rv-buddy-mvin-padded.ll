@.str = private unnamed_addr constant [12 x i8] c"config_ld.\0A\00", align 1
@.str.1 = private unnamed_addr constant [12 x i8] c"config_st.\0A\00", align 1
@.str.2 = private unnamed_addr constant [7 x i8] c"mvin.\0A\00", align 1
@.str.3 = private unnamed_addr constant [8 x i8] c"mvout.\0A\00", align 1
@.str.4 = private unnamed_addr constant [20 x i8] c"print input array.\0A\00", align 1
@.str.5 = private unnamed_addr constant [4 x i8] c"%d \00", align 1
@.str.6 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@.str.7 = private unnamed_addr constant [21 x i8] c"print output array.\0A\00", align 1

define dso_local signext i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %input = alloca [13 x [14 x i8]], align 1
  %output = alloca [16 x [16 x i8]], align 1
  %i = alloca i64, align 8
  %j = alloca i64, align 8
  %inputAddr = alloca i64, align 8
  %outputAddr = alloca i64, align 8
  %i14 = alloca i64, align 8
  %j19 = alloca i64, align 8
  %i36 = alloca i64, align 8
  %j41 = alloca i64, align 8
  store i32 0, ptr %retval, align 4
  store i64 0, ptr %i, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.inc5, %entry
  %0 = load i64, ptr %i, align 8
  %cmp = icmp ult i64 %0, 13
  br i1 %cmp, label %for.body, label %for.end7

for.body:                                         ; preds = %for.cond
  store i64 0, ptr %j, align 8
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %1 = load i64, ptr %j, align 8
  %cmp2 = icmp ult i64 %1, 14
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %call = call signext i32 @rand()
  %rem = srem i32 %call, 127
  %conv = trunc i32 %rem to i8
  %2 = load i64, ptr %i, align 8
  %arrayidx = getelementptr inbounds [13 x [14 x i8]], ptr %input, i64 0, i64 %2
  %3 = load i64, ptr %j, align 8
  %arrayidx4 = getelementptr inbounds [14 x i8], ptr %arrayidx, i64 0, i64 %3
  store i8 %conv, ptr %arrayidx4, align 1
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %4 = load i64, ptr %j, align 8
  %inc = add i64 %4, 1
  store i64 %inc, ptr %j, align 8
  br label %for.cond1, !llvm.loop !5

for.end:                                          ; preds = %for.cond1
  br label %for.inc5

for.inc5:                                         ; preds = %for.end
  %5 = load i64, ptr %i, align 8
  %inc6 = add i64 %5, 1
  store i64 %inc6, ptr %i, align 8
  br label %for.cond, !llvm.loop !7

for.end7:                                         ; preds = %for.cond
  %arraydecay = getelementptr inbounds [13 x [14 x i8]], ptr %input, i64 0, i64 0
  %6 = ptrtoint ptr %arraydecay to i64
  store i64 %6, ptr %inputAddr, align 8
  %arraydecay8 = getelementptr inbounds [16 x [16 x i8]], ptr %output, i64 0, i64 0
  %7 = ptrtoint ptr %arraydecay8 to i64
  store i64 %7, ptr %outputAddr, align 8
  call void @llvm.riscv.configLd(i64 4575657221409472769, i64 14)
  %call9 = call signext i32 (ptr, ...) @printf(ptr noundef @.str) 
  call void @llvm.riscv.configSt(i64 2, i64 4575657221408423952)
  %call10 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.1)
  call void @llvm.riscv.mvin(i64 %6, i64 3659234826780672)
  %call11 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.2)
  call void @llvm.riscv.mvout(i64 %7, i64 4503668346847232)
  %call12 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.3)
  %call13 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.4)
  store i64 0, ptr %i14, align 8
  br label %for.cond15

for.cond15:                                       ; preds = %for.inc32, %for.end7
  %8 = load i64, ptr %i14, align 8
  %cmp16 = icmp ult i64 %8, 13
  br i1 %cmp16, label %for.body18, label %for.end34

for.body18:                                       ; preds = %for.cond15
  store i64 0, ptr %j19, align 8
  br label %for.cond20

for.cond20:                                       ; preds = %for.inc28, %for.body18
  %9 = load i64, ptr %j19, align 8
  %cmp21 = icmp ult i64 %9, 14
  br i1 %cmp21, label %for.body23, label %for.end30

for.body23:                                       ; preds = %for.cond20
  %10 = load i64, ptr %i14, align 8
  %arrayidx24 = getelementptr inbounds [13 x [14 x i8]], ptr %input, i64 0, i64 %10
  %11 = load i64, ptr %j19, align 8
  %arrayidx25 = getelementptr inbounds [14 x i8], ptr %arrayidx24, i64 0, i64 %11
  %12 = load i8, ptr %arrayidx25, align 1
  %conv26 = sext i8 %12 to i32
  %call27 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.5, i32 noundef signext %conv26)
  br label %for.inc28

for.inc28:                                        ; preds = %for.body23
  %13 = load i64, ptr %j19, align 8
  %inc29 = add i64 %13, 1
  store i64 %inc29, ptr %j19, align 8
  br label %for.cond20, !llvm.loop !8

for.end30:                                        ; preds = %for.cond20
  %call31 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.6)
  br label %for.inc32

for.inc32:                                        ; preds = %for.end30
  %14 = load i64, ptr %i14, align 8
  %inc33 = add i64 %14, 1
  store i64 %inc33, ptr %i14, align 8
  br label %for.cond15, !llvm.loop !9

for.end34:                                        ; preds = %for.cond15
  %call35 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.7)
  store i64 0, ptr %i36, align 8
  br label %for.cond37

for.cond37:                                       ; preds = %for.inc54, %for.end34
  %15 = load i64, ptr %i36, align 8
  %cmp38 = icmp ult i64 %15, 16
  br i1 %cmp38, label %for.body40, label %for.end56

for.body40:                                       ; preds = %for.cond37
  store i64 0, ptr %j41, align 8
  br label %for.cond42

for.cond42:                                       ; preds = %for.inc50, %for.body40
  %16 = load i64, ptr %j41, align 8
  %cmp43 = icmp ult i64 %16, 16
  br i1 %cmp43, label %for.body45, label %for.end52

for.body45:                                       ; preds = %for.cond42
  %17 = load i64, ptr %i36, align 8
  %arrayidx46 = getelementptr inbounds [16 x [16 x i8]], ptr %output, i64 0, i64 %17
  %18 = load i64, ptr %j41, align 8
  %arrayidx47 = getelementptr inbounds [16 x i8], ptr %arrayidx46, i64 0, i64 %18
  %19 = load i8, ptr %arrayidx47, align 1
  %conv48 = sext i8 %19 to i32
  %call49 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.5, i32 noundef signext %conv48)
  br label %for.inc50

for.inc50:                                        ; preds = %for.body45
  %20 = load i64, ptr %j41, align 8
  %inc51 = add i64 %20, 1
  store i64 %inc51, ptr %j41, align 8
  br label %for.cond42, !llvm.loop !10

for.end52:                                        ; preds = %for.cond42
  %call53 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.6)
  br label %for.inc54

for.inc54:                                        ; preds = %for.end52
  %21 = load i64, ptr %i36, align 8
  %inc55 = add i64 %21, 1
  store i64 %inc55, ptr %i36, align 8
  br label %for.cond37, !llvm.loop !11

for.end56:                                        ; preds = %for.cond37
  %22 = load i32, ptr %retval, align 4
  ret i32 %22
}

declare dso_local signext i32 @rand() #1
declare dso_local signext i32 @printf(ptr noundef, ...) #1
declare void @llvm.riscv.mvin(i64, i64)
declare void @llvm.riscv.mvout(i64, i64)
declare void @llvm.riscv.flush(i64, i64)
declare void @llvm.riscv.configSt(i64, i64)
declare void @llvm.riscv.configLd(i64, i64)

attributes #0 = { noinline nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+m,+relax,-save-restore" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+m,+relax,-save-restore" }

!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.mustprogress"}
!7 = distinct !{!7, !6}
!8 = distinct !{!8, !6}
!9 = distinct !{!9, !6}
!10 = distinct !{!10, !6}
!11 = distinct !{!11, !6}
