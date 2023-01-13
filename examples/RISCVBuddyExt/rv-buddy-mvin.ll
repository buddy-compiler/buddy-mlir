define dso_local signext i32 @main() {
entry:
  %retval = alloca i32, align 4
  %array = alloca [2 x [3 x i8]], align 1
  %n = alloca i32, align 4
  %m = alloca i32, align 4
  %addr = alloca i64, align 8
  store i32 0, ptr %retval, align 4
  store i32 0, ptr %n, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc6, %entry
  %0 = load i32, ptr %n, align 4
  %cmp = icmp slt i32 %0, 2
  br i1 %cmp, label %for.body, label %for.end8

for.body:                                         ; preds = %for.cond
  store i32 0, ptr %m, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %1 = load i32, ptr %m, align 4
  %cmp2 = icmp slt i32 %1, 3
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %2 = load i32, ptr %n, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds [2 x [3 x i8]], ptr %array, i64 0, i64 %idxprom
  %3 = load i32, ptr %m, align 4
  %idxprom4 = sext i32 %3 to i64
  %arrayidx5 = getelementptr inbounds [3 x i8], ptr %arrayidx, i64 0, i64 %idxprom4
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
  call void @llvm.riscv.flush(i64 0, i64 0)
  %arraydecay = getelementptr inbounds [2 x [3 x i8]], ptr %array, i64 0, i64 0
  %6 = ptrtoint ptr %arraydecay to i64
  store i64 %6, ptr %addr, align 8
  call void @llvm.riscv.mvin(i64 %6, i64 4503668346847232)
  ret i32 0
}

declare void @llvm.riscv.mvin(i64, i64)
declare void @llvm.riscv.flush(i64, i64)

!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.mustprogress"}
!7 = distinct !{!7, !6}
