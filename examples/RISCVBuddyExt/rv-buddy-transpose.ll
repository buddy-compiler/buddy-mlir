@.str = private unnamed_addr constant [45 x i8] c"Flush Gemini TLB of stale virtual addresses\0A\00", align 1
@.str.1 = private unnamed_addr constant [57 x i8] c"Initialize our input and output matrices in main memory\0A\00", align 1
@.str.2 = private unnamed_addr constant [56 x i8] c"Calculate the scratchpad addresses of all our matrices\0A\00", align 1
@.str.3 = private unnamed_addr constant [87 x i8] c"  Note: The scratchpad is \22row-addressed\22, where each address contains one matrix row\0A\00", align 1
@.str.4 = private unnamed_addr constant [61 x i8] c"Move \22In\22 matrix from main memory into Gemmini's scratchpad\0A\00", align 1
@.str.5 = private unnamed_addr constant [67 x i8] c"Move \22Identity\22 matrix from main memory into Gemmini's scratchpad\0A\00", align 1
@.str.6 = private unnamed_addr constant [62 x i8] c"Multiply \22In\22 matrix with \22Identity\22 matrix with a bias of 0\0A\00", align 1
@.str.7 = private unnamed_addr constant [62 x i8] c"Move \22Out\22 matrix from Gemmini's scratchpad into main memory\0A\00", align 1
@.str.8 = private unnamed_addr constant [16 x i8] c"print arrayIn.\0A\00", align 1
@.str.9 = private unnamed_addr constant [4 x i8] c"%d \00", align 1
@.str.10 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@.str.11 = private unnamed_addr constant [17 x i8] c"print arrayOut.\0A\00", align 1

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %In = alloca [16 x [16 x i8]], align 1
  %Out = alloca [16 x [16 x i8]], align 1
  %Identity = alloca [16 x [16 x i8]], align 1
  %InAddr = alloca i64, align 8
  %OutAddr = alloca i64, align 8
  %IdentityAddr = alloca i64, align 8
  %i = alloca i64, align 8
  %j = alloca i64, align 8
  %In_sp_addr = alloca i64, align 8
  %Out_sp_addr = alloca i64, align 8
  %Indentity_sp_addr = alloca i64, align 8
  %i24 = alloca i64, align 8
  %j29 = alloca i64, align 8
  %i46 = alloca i64, align 8
  %j51 = alloca i64, align 8
  store i32 0, ptr %retval, align 4
  %call = call signext i32 (ptr, ...) @printf(ptr noundef @.str)
  %call1 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.1)
  %arraydecay = getelementptr inbounds [16 x [16 x i8]], ptr %In, i64 0, i64 0
  %0 = ptrtoint ptr %arraydecay to i64
  store i64 %0, ptr %InAddr, align 8
  %arraydecay2 = getelementptr inbounds [16 x [16 x i8]], ptr %Out, i64 0, i64 0
  %1 = ptrtoint ptr %arraydecay2 to i64
  store i64 %1, ptr %OutAddr, align 8
  %arraydecay3 = getelementptr inbounds [16 x [16 x i8]], ptr %Identity, i64 0, i64 0
  %2 = ptrtoint ptr %arraydecay3 to i64
  store i64 %2, ptr %IdentityAddr, align 8
  store i64 0, ptr %i, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.inc14, %entry
  %3 = load i64, ptr %i, align 8
  %cmp = icmp ult i64 %3, 16
  br i1 %cmp, label %for.body, label %for.end16

for.body:                                         ; preds = %for.cond
  store i64 0, ptr %j, align 8
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc, %for.body
  %4 = load i64, ptr %j, align 8
  %cmp5 = icmp ult i64 %4, 16
  br i1 %cmp5, label %for.body6, label %for.end

for.body6:                                        ; preds = %for.cond4
  %call7 = call signext i32 @rand()
  %rem = srem i32 %call7, 10
  %conv = trunc i32 %rem to i8
  %5 = load i64, ptr %i, align 8
  %arrayidx = getelementptr inbounds [16 x [16 x i8]], ptr %In, i64 0, i64 %5
  %6 = load i64, ptr %j, align 8
  %arrayidx8 = getelementptr inbounds [16 x i8], ptr %arrayidx, i64 0, i64 %6
  store i8 %conv, ptr %arrayidx8, align 1
  %7 = load i64, ptr %i, align 8
  %8 = load i64, ptr %j, align 8
  %cmp9 = icmp eq i64 %7, %8
  %conv10 = zext i1 %cmp9 to i32
  %conv11 = trunc i32 %conv10 to i8
  %9 = load i64, ptr %i, align 8
  %arrayidx12 = getelementptr inbounds [16 x [16 x i8]], ptr %Identity, i64 0, i64 %9
  %10 = load i64, ptr %j, align 8
  %arrayidx13 = getelementptr inbounds [16 x i8], ptr %arrayidx12, i64 0, i64 %10
  store i8 %conv11, ptr %arrayidx13, align 1
  br label %for.inc

for.inc:                                          ; preds = %for.body6
  %11 = load i64, ptr %j, align 8
  %inc = add i64 %11, 1
  store i64 %inc, ptr %j, align 8
  br label %for.cond4, !llvm.loop !5

for.end:                                          ; preds = %for.cond4
  br label %for.inc14

for.inc14:                                        ; preds = %for.end
  %12 = load i64, ptr %i, align 8
  %inc15 = add i64 %12, 1
  store i64 %inc15, ptr %i, align 8
  br label %for.cond, !llvm.loop !7

for.end16:                                        ; preds = %for.cond
  %call17 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.2)
  %call18 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.3)
  store i64 0, ptr %In_sp_addr, align 8
  store i64 16, ptr %Out_sp_addr, align 8
  store i64 32, ptr %Indentity_sp_addr, align 8
  %call19 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.4)
  call void @llvm.riscv.configLd(i64 4575657221409472769, i64 16)
  call void @llvm.riscv.configSt(i64 2, i64 4575657221408423952)
  call void @llvm.riscv.mvin(i64 %0, i64 4503668346847232)
  %call20 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.5)
  call void @llvm.riscv.mvin(i64 %2, i64 4503668346847264)
  %call21 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.6)
  call void @llvm.riscv.configEx(i64 4575657221408489728, i64 281474976710656)
  call void @llvm.riscv.preload(i64 4503672641814527, i64 4503668346847248)
  call void @llvm.riscv.computePreloaded(i64 4503668346847232, i64 4503668346847264)
  %call22 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.7)
  call void @llvm.riscv.configSt(i64 2, i64 4575657221408423952)
  call void @llvm.riscv.mvout(i64 %1, i64 4503668346847248)
  %call23 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.8)
  store i64 0, ptr %i24, align 8
  br label %for.cond25

for.cond25:                                       ; preds = %for.inc42, %for.end16
  %13 = load i64, ptr %i24, align 8
  %cmp26 = icmp ult i64 %13, 16
  br i1 %cmp26, label %for.body28, label %for.end44

for.body28:                                       ; preds = %for.cond25
  store i64 0, ptr %j29, align 8
  br label %for.cond30

for.cond30:                                       ; preds = %for.inc38, %for.body28
  %14 = load i64, ptr %j29, align 8
  %cmp31 = icmp ult i64 %14, 16
  br i1 %cmp31, label %for.body33, label %for.end40

for.body33:                                       ; preds = %for.cond30
  %15 = load i64, ptr %i24, align 8
  %arrayidx34 = getelementptr inbounds [16 x [16 x i8]], ptr %In, i64 0, i64 %15
  %16 = load i64, ptr %j29, align 8
  %arrayidx35 = getelementptr inbounds [16 x i8], ptr %arrayidx34, i64 0, i64 %16
  %17 = load i8, ptr %arrayidx35, align 1
  %conv36 = sext i8 %17 to i32
  %call37 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.9, i32 noundef signext %conv36)
  br label %for.inc38

for.inc38:                                        ; preds = %for.body33
  %18 = load i64, ptr %j29, align 8
  %inc39 = add i64 %18, 1
  store i64 %inc39, ptr %j29, align 8
  br label %for.cond30, !llvm.loop !8

for.end40:                                        ; preds = %for.cond30
  %call41 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.10)
  br label %for.inc42

for.inc42:                                        ; preds = %for.end40
  %19 = load i64, ptr %i24, align 8
  %inc43 = add i64 %19, 1
  store i64 %inc43, ptr %i24, align 8
  br label %for.cond25, !llvm.loop !9

for.end44:                                        ; preds = %for.cond25
  %call45 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.11)
  store i64 0, ptr %i46, align 8
  br label %for.cond47

for.cond47:                                       ; preds = %for.inc64, %for.end44
  %20 = load i64, ptr %i46, align 8
  %cmp48 = icmp ult i64 %20, 16
  br i1 %cmp48, label %for.body50, label %for.end66

for.body50:                                       ; preds = %for.cond47
  store i64 0, ptr %j51, align 8
  br label %for.cond52

for.cond52:                                       ; preds = %for.inc60, %for.body50
  %21 = load i64, ptr %j51, align 8
  %cmp53 = icmp ult i64 %21, 16
  br i1 %cmp53, label %for.body55, label %for.end62

for.body55:                                       ; preds = %for.cond52
  %22 = load i64, ptr %i46, align 8
  %arrayidx56 = getelementptr inbounds [16 x [16 x i8]], ptr %Out, i64 0, i64 %22
  %23 = load i64, ptr %j51, align 8
  %arrayidx57 = getelementptr inbounds [16 x i8], ptr %arrayidx56, i64 0, i64 %23
  %24 = load i8, ptr %arrayidx57, align 1
  %conv58 = sext i8 %24 to i32
  %call59 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.9, i32 noundef signext %conv58)
  br label %for.inc60

for.inc60:                                        ; preds = %for.body55
  %25 = load i64, ptr %j51, align 8
  %inc61 = add i64 %25, 1
  store i64 %inc61, ptr %j51, align 8
  br label %for.cond52, !llvm.loop !10

for.end62:                                        ; preds = %for.cond52
  %call63 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.10)
  br label %for.inc64

for.inc64:                                        ; preds = %for.end62
  %26 = load i64, ptr %i46, align 8
  %inc65 = add i64 %26, 1
  store i64 %inc65, ptr %i46, align 8
  br label %for.cond47, !llvm.loop !11

for.end66:                                        ; preds = %for.cond47
  ret i32 0
}

declare dso_local signext i32 @printf(ptr noundef, ...) #1
declare dso_local signext i32 @rand() #1
declare void @llvm.riscv.mvin(i64, i64)
declare void @llvm.riscv.mvout(i64, i64)
declare void @llvm.riscv.flush(i64, i64)
declare void @llvm.riscv.configLd(i64, i64)
declare void @llvm.riscv.configSt(i64, i64)
declare void @llvm.riscv.configEx(i64, i64)
declare void @llvm.riscv.preload(i64, i64)
declare void @llvm.riscv.computePreloaded(i64, i64)

attributes #0 = { noinline nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+m,+relax,-save-restore" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+m,+relax,-save-restore" }

!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.mustprogress"}
!7 = distinct !{!7, !6}
!8 = distinct !{!8, !6}
!9 = distinct !{!9, !6}
!10 = distinct !{!10, !6}
!11 = distinct !{!11, !6}
