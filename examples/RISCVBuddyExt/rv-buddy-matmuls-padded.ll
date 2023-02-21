@.str = private unnamed_addr constant [13 x i8] c"gold array.\0A\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"%d \00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@.str.3 = private unnamed_addr constant [31 x i8] c"dataflow = OUTPUT_STATIONARY.\0A\00", align 1
@.str.4 = private unnamed_addr constant [12 x i8] c"config_ex.\0A\00", align 1
@.str.5 = private unnamed_addr constant [12 x i8] c"config_st.\0A\00", align 1
@.str.6 = private unnamed_addr constant [12 x i8] c"config_ld.\0A\00", align 1
@.str.7 = private unnamed_addr constant [7 x i8] c"mvin.\0A\00", align 1
@.str.8 = private unnamed_addr constant [10 x i8] c"preload.\0A\00", align 1
@.str.9 = private unnamed_addr constant [20 x i8] c"compute_preloaded.\0A\00", align 1
@.str.10 = private unnamed_addr constant [8 x i8] c"mvout.\0A\00", align 1
@.str.11 = private unnamed_addr constant [17 x i8] c"compute result.\0A\00", align 1
@.str.12 = private unnamed_addr constant [31 x i8] c"dataflow = WEIGHT_STATIONARY.\0A\00", align 1

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %I = alloca i64, align 8
  %J = alloca i64, align 8
  %K = alloca i64, align 8
  %saved_stack = alloca ptr, align 8
  %__vla_expr0 = alloca i64, align 8
  %__vla_expr1 = alloca i64, align 8
  %__vla_expr2 = alloca i64, align 8
  %__vla_expr3 = alloca i64, align 8
  %__vla_expr4 = alloca i64, align 8
  %__vla_expr5 = alloca i64, align 8
  %__vla_expr6 = alloca i64, align 8
  %__vla_expr7 = alloca i64, align 8
  %__vla_expr8 = alloca i64, align 8
  %__vla_expr9 = alloca i64, align 8
  %i = alloca i64, align 8
  %k = alloca i64, align 8
  %k12 = alloca i64, align 8
  %j = alloca i64, align 8
  %i32 = alloca i64, align 8
  %j37 = alloca i64, align 8
  %i53 = alloca i64, align 8
  %j58 = alloca i64, align 8
  %result = alloca i32, align 4
  %k66 = alloca i64, align 8
  %i98 = alloca i64, align 8
  %j103 = alloca i64, align 8
  %Aaddr = alloca i64, align 8
  %Baddr = alloca i64, align 8
  %Caddr = alloca i64, align 8
  %Daddr = alloca i64, align 8
  %i132 = alloca i64, align 8
  %j137 = alloca i64, align 8
  %i166 = alloca i64, align 8
  %j171 = alloca i64, align 8
  store i32 0, ptr %retval, align 4
  store i64 14, ptr %I, align 8
  store i64 15, ptr %J, align 8
  store i64 16, ptr %K, align 8
  %0 = load i64, ptr %I, align 8
  %1 = load i64, ptr %K, align 8
  %2 = call ptr @llvm.stacksave()
  store ptr %2, ptr %saved_stack, align 8
  %3 = mul nuw i64 %0, %1
  %vla = alloca i8, i64 %3, align 1
  store i64 %0, ptr %__vla_expr0, align 8
  store i64 %1, ptr %__vla_expr1, align 8
  %4 = load i64, ptr %K, align 8
  %5 = load i64, ptr %J, align 8
  %6 = mul nuw i64 %4, %5
  %vla1 = alloca i8, i64 %6, align 1
  store i64 %4, ptr %__vla_expr2, align 8
  store i64 %5, ptr %__vla_expr3, align 8
  %7 = load i64, ptr %I, align 8
  %8 = load i64, ptr %J, align 8
  %9 = mul nuw i64 %7, %8
  %vla2 = alloca i8, i64 %9, align 1
  store i64 %7, ptr %__vla_expr4, align 8
  store i64 %8, ptr %__vla_expr5, align 8
  %10 = load i64, ptr %I, align 8
  %11 = load i64, ptr %J, align 8
  %12 = mul nuw i64 %10, %11
  %vla3 = alloca i8, i64 %12, align 1
  store i64 %10, ptr %__vla_expr6, align 8
  store i64 %11, ptr %__vla_expr7, align 8
  %13 = load i64, ptr %I, align 8
  %14 = load i64, ptr %J, align 8
  %15 = mul nuw i64 %13, %14
  %vla4 = alloca i8, i64 %15, align 1
  store i64 %13, ptr %__vla_expr8, align 8
  store i64 %14, ptr %__vla_expr9, align 8
  store i64 0, ptr %i, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.inc9, %entry
  %16 = load i64, ptr %i, align 8
  %17 = load i64, ptr %I, align 8
  %cmp = icmp ult i64 %16, %17
  br i1 %cmp, label %for.body, label %for.end11

for.body:                                         ; preds = %for.cond
  store i64 0, ptr %k, align 8
  br label %for.cond5

for.cond5:                                        ; preds = %for.inc, %for.body
  %18 = load i64, ptr %k, align 8
  %19 = load i64, ptr %K, align 8
  %cmp6 = icmp ult i64 %18, %19
  br i1 %cmp6, label %for.body7, label %for.end

for.body7:                                        ; preds = %for.cond5
  %call = call signext i32 @rand()
  %rem = srem i32 %call, 5
  %conv = trunc i32 %rem to i8
  %20 = load i64, ptr %i, align 8
  %21 = mul nsw i64 %20, %1
  %arrayidx = getelementptr inbounds i8, ptr %vla, i64 %21
  %22 = load i64, ptr %k, align 8
  %arrayidx8 = getelementptr inbounds i8, ptr %arrayidx, i64 %22
  store i8 %conv, ptr %arrayidx8, align 1
  br label %for.inc

for.inc:                                          ; preds = %for.body7
  %23 = load i64, ptr %k, align 8
  %inc = add i64 %23, 1
  store i64 %inc, ptr %k, align 8
  br label %for.cond5, !llvm.loop !5

for.end:                                          ; preds = %for.cond5
  br label %for.inc9

for.inc9:                                         ; preds = %for.end
  %24 = load i64, ptr %i, align 8
  %inc10 = add i64 %24, 1
  store i64 %inc10, ptr %i, align 8
  br label %for.cond, !llvm.loop !7

for.end11:                                        ; preds = %for.cond
  store i64 0, ptr %k12, align 8
  br label %for.cond13

for.cond13:                                       ; preds = %for.inc29, %for.end11
  %25 = load i64, ptr %k12, align 8
  %26 = load i64, ptr %K, align 8
  %cmp14 = icmp ult i64 %25, %26
  br i1 %cmp14, label %for.body16, label %for.end31

for.body16:                                       ; preds = %for.cond13
  store i64 0, ptr %j, align 8
  br label %for.cond17

for.cond17:                                       ; preds = %for.inc26, %for.body16
  %27 = load i64, ptr %j, align 8
  %28 = load i64, ptr %J, align 8
  %cmp18 = icmp ult i64 %27, %28
  br i1 %cmp18, label %for.body20, label %for.end28

for.body20:                                       ; preds = %for.cond17
  %call21 = call signext i32 @rand()
  %rem22 = srem i32 %call21, 5
  %conv23 = trunc i32 %rem22 to i8
  %29 = load i64, ptr %k12, align 8
  %30 = mul nsw i64 %29, %5
  %arrayidx24 = getelementptr inbounds i8, ptr %vla1, i64 %30
  %31 = load i64, ptr %j, align 8
  %arrayidx25 = getelementptr inbounds i8, ptr %arrayidx24, i64 %31
  store i8 %conv23, ptr %arrayidx25, align 1
  br label %for.inc26

for.inc26:                                        ; preds = %for.body20
  %32 = load i64, ptr %j, align 8
  %inc27 = add i64 %32, 1
  store i64 %inc27, ptr %j, align 8
  br label %for.cond17, !llvm.loop !8

for.end28:                                        ; preds = %for.cond17
  br label %for.inc29

for.inc29:                                        ; preds = %for.end28
  %33 = load i64, ptr %k12, align 8
  %inc30 = add i64 %33, 1
  store i64 %inc30, ptr %k12, align 8
  br label %for.cond13, !llvm.loop !9

for.end31:                                        ; preds = %for.cond13
  store i64 0, ptr %i32, align 8
  br label %for.cond33

for.cond33:                                       ; preds = %for.inc50, %for.end31
  %34 = load i64, ptr %i32, align 8
  %35 = load i64, ptr %I, align 8
  %cmp34 = icmp ult i64 %34, %35
  br i1 %cmp34, label %for.body36, label %for.end52

for.body36:                                       ; preds = %for.cond33
  store i64 0, ptr %j37, align 8
  br label %for.cond38

for.cond38:                                       ; preds = %for.inc47, %for.body36
  %36 = load i64, ptr %j37, align 8
  %37 = load i64, ptr %J, align 8
  %cmp39 = icmp ult i64 %36, %37
  br i1 %cmp39, label %for.body41, label %for.end49

for.body41:                                       ; preds = %for.cond38
  %call42 = call signext i32 @rand()
  %rem43 = srem i32 %call42, 5
  %conv44 = trunc i32 %rem43 to i8
  %38 = load i64, ptr %i32, align 8
  %39 = mul nsw i64 %38, %8
  %arrayidx45 = getelementptr inbounds i8, ptr %vla2, i64 %39
  %40 = load i64, ptr %j37, align 8
  %arrayidx46 = getelementptr inbounds i8, ptr %arrayidx45, i64 %40
  store i8 %conv44, ptr %arrayidx46, align 1
  br label %for.inc47

for.inc47:                                        ; preds = %for.body41
  %41 = load i64, ptr %j37, align 8
  %inc48 = add i64 %41, 1
  store i64 %inc48, ptr %j37, align 8
  br label %for.cond38, !llvm.loop !10

for.end49:                                        ; preds = %for.cond38
  br label %for.inc50

for.inc50:                                        ; preds = %for.end49
  %42 = load i64, ptr %i32, align 8
  %inc51 = add i64 %42, 1
  store i64 %inc51, ptr %i32, align 8
  br label %for.cond33, !llvm.loop !11

for.end52:                                        ; preds = %for.cond33
  store i64 0, ptr %i53, align 8
  br label %for.cond54

for.cond54:                                       ; preds = %for.inc94, %for.end52
  %43 = load i64, ptr %i53, align 8
  %44 = load i64, ptr %I, align 8
  %cmp55 = icmp ult i64 %43, %44
  br i1 %cmp55, label %for.body57, label %for.end96

for.body57:                                       ; preds = %for.cond54
  store i64 0, ptr %j58, align 8
  br label %for.cond59

for.cond59:                                       ; preds = %for.inc91, %for.body57
  %45 = load i64, ptr %j58, align 8
  %46 = load i64, ptr %J, align 8
  %cmp60 = icmp ult i64 %45, %46
  br i1 %cmp60, label %for.body62, label %for.end93

for.body62:                                       ; preds = %for.cond59
  %47 = load i64, ptr %i53, align 8
  %48 = mul nsw i64 %47, %8
  %arrayidx63 = getelementptr inbounds i8, ptr %vla2, i64 %48
  %49 = load i64, ptr %j58, align 8
  %arrayidx64 = getelementptr inbounds i8, ptr %arrayidx63, i64 %49
  %50 = load i8, ptr %arrayidx64, align 1
  %conv65 = sext i8 %50 to i32
  store i32 %conv65, ptr %result, align 4
  store i64 0, ptr %k66, align 8
  br label %for.cond67

for.cond67:                                       ; preds = %for.inc77, %for.body62
  %51 = load i64, ptr %k66, align 8
  %52 = load i64, ptr %K, align 8
  %cmp68 = icmp ult i64 %51, %52
  br i1 %cmp68, label %for.body70, label %for.end79

for.body70:                                       ; preds = %for.cond67
  %53 = load i64, ptr %i53, align 8
  %54 = mul nsw i64 %53, %1
  %arrayidx71 = getelementptr inbounds i8, ptr %vla, i64 %54
  %55 = load i64, ptr %k66, align 8
  %arrayidx72 = getelementptr inbounds i8, ptr %arrayidx71, i64 %55
  %56 = load i8, ptr %arrayidx72, align 1
  %conv73 = sext i8 %56 to i32
  %57 = load i64, ptr %k66, align 8
  %58 = mul nsw i64 %57, %5
  %arrayidx74 = getelementptr inbounds i8, ptr %vla1, i64 %58
  %59 = load i64, ptr %j58, align 8
  %arrayidx75 = getelementptr inbounds i8, ptr %arrayidx74, i64 %59
  %60 = load i8, ptr %arrayidx75, align 1
  %conv76 = sext i8 %60 to i32
  %mul = mul nsw i32 %conv73, %conv76
  %61 = load i32, ptr %result, align 4
  %add = add nsw i32 %61, %mul
  store i32 %add, ptr %result, align 4
  br label %for.inc77

for.inc77:                                        ; preds = %for.body70
  %62 = load i64, ptr %k66, align 8
  %inc78 = add i64 %62, 1
  store i64 %inc78, ptr %k66, align 8
  br label %for.cond67, !llvm.loop !12

for.end79:                                        ; preds = %for.cond67
  %63 = load i32, ptr %result, align 4
  %cmp80 = icmp slt i32 %63, -128
  br i1 %cmp80, label %cond.true, label %cond.false

cond.true:                                        ; preds = %for.end79
  br label %cond.end86

cond.false:                                       ; preds = %for.end79
  %64 = load i32, ptr %result, align 4
  %cmp82 = icmp sgt i32 %64, 127
  br i1 %cmp82, label %cond.true84, label %cond.false85

cond.true84:                                      ; preds = %cond.false
  br label %cond.end

cond.false85:                                     ; preds = %cond.false
  %65 = load i32, ptr %result, align 4
  br label %cond.end

cond.end:                                         ; preds = %cond.false85, %cond.true84
  %cond = phi i32 [ 127, %cond.true84 ], [ %65, %cond.false85 ]
  br label %cond.end86

cond.end86:                                       ; preds = %cond.end, %cond.true
  %cond87 = phi i32 [ -128, %cond.true ], [ %cond, %cond.end ]
  %conv88 = trunc i32 %cond87 to i8
  %66 = load i64, ptr %i53, align 8
  %67 = mul nsw i64 %66, %14
  %arrayidx89 = getelementptr inbounds i8, ptr %vla4, i64 %67
  %68 = load i64, ptr %j58, align 8
  %arrayidx90 = getelementptr inbounds i8, ptr %arrayidx89, i64 %68
  store i8 %conv88, ptr %arrayidx90, align 1
  br label %for.inc91

for.inc91:                                        ; preds = %cond.end86
  %69 = load i64, ptr %j58, align 8
  %inc92 = add i64 %69, 1
  store i64 %inc92, ptr %j58, align 8
  br label %for.cond59, !llvm.loop !13

for.end93:                                        ; preds = %for.cond59
  br label %for.inc94

for.inc94:                                        ; preds = %for.end93
  %70 = load i64, ptr %i53, align 8
  %inc95 = add i64 %70, 1
  store i64 %inc95, ptr %i53, align 8
  br label %for.cond54, !llvm.loop !14

for.end96:                                        ; preds = %for.cond54
  %call97 = call signext i32 (ptr, ...) @printf(ptr noundef @.str)
  store i64 0, ptr %i98, align 8
  br label %for.cond99

for.cond99:                                       ; preds = %for.inc116, %for.end96
  %71 = load i64, ptr %i98, align 8
  %72 = load i64, ptr %I, align 8
  %cmp100 = icmp ult i64 %71, %72
  br i1 %cmp100, label %for.body102, label %for.end118

for.body102:                                      ; preds = %for.cond99
  store i64 0, ptr %j103, align 8
  br label %for.cond104

for.cond104:                                      ; preds = %for.inc112, %for.body102
  %73 = load i64, ptr %j103, align 8
  %74 = load i64, ptr %J, align 8
  %cmp105 = icmp ult i64 %73, %74
  br i1 %cmp105, label %for.body107, label %for.end114

for.body107:                                      ; preds = %for.cond104
  %75 = load i64, ptr %i98, align 8
  %76 = mul nsw i64 %75, %14
  %arrayidx108 = getelementptr inbounds i8, ptr %vla4, i64 %76
  %77 = load i64, ptr %j103, align 8
  %arrayidx109 = getelementptr inbounds i8, ptr %arrayidx108, i64 %77
  %78 = load i8, ptr %arrayidx109, align 1
  %conv110 = sext i8 %78 to i32
  %call111 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef signext %conv110)
  br label %for.inc112

for.inc112:                                       ; preds = %for.body107
  %79 = load i64, ptr %j103, align 8
  %inc113 = add i64 %79, 1
  store i64 %inc113, ptr %j103, align 8
  br label %for.cond104, !llvm.loop !15

for.end114:                                       ; preds = %for.cond104
  %call115 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.2)
  br label %for.inc116

for.inc116:                                       ; preds = %for.end114
  %80 = load i64, ptr %i98, align 8
  %inc117 = add i64 %80, 1
  store i64 %inc117, ptr %i98, align 8
  br label %for.cond99, !llvm.loop !16

for.end118:                                       ; preds = %for.cond99
  %81 = ptrtoint ptr %vla to i64
  store i64 %81, ptr %Aaddr, align 8
  %82 = ptrtoint ptr %vla1 to i64
  store i64 %82, ptr %Baddr, align 8
  %83 = ptrtoint ptr %vla3 to i64
  store i64 %83, ptr %Caddr, align 8
  %84 = ptrtoint ptr %vla2 to i64
  store i64 %84, ptr %Daddr, align 8
  %call119 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.3)
  call void @llvm.riscv.configEx(i64 4575657221408489472, i64 281474976710656)
  %call120 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.4)
  call void @llvm.riscv.configSt(i64 2, i64 4575657221408423951)
  %call121 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.5)
  call void @llvm.riscv.configLd(i64 4575657221409472769, i64 16)
  %call122 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.6)
  call void @llvm.riscv.mvin(i64 %81, i64 3940718393425920)
  %call123 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.7)
  call void @llvm.riscv.configLd(i64 4575657221409472769, i64 15)
  %call124 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.6)
  call void @llvm.riscv.mvin(i64 %82, i64 4503664051879952)
  %call125 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.7)
  call void @llvm.riscv.configLd(i64 4575657221409472769, i64 15)
  %call126 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.6)
  call void @llvm.riscv.mvin(i64 %84, i64 3940714098458656)
  %call127 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.7)
  call void @llvm.riscv.preload(i64 3940714098458656, i64 3940714098458672)
  %call128 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.8)
   call void @llvm.riscv.computePreloaded(i64 3940718393425920, i64 4503664051879952)
  %call129 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.9)
  call void @llvm.riscv.mvout(i64 %83, i64 3940714098458672)
  %call130 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.10)
  %call131 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.11)
  call void asm sideeffect "fence", ""()
  store i64 0, ptr %i132, align 8
  br label %for.cond133

for.cond133:                                      ; preds = %for.inc150, %for.end118
  %85 = load i64, ptr %i132, align 8
  %86 = load i64, ptr %I, align 8
  %cmp134 = icmp ult i64 %85, %86
  br i1 %cmp134, label %for.body136, label %for.end152

for.body136:                                      ; preds = %for.cond133
  store i64 0, ptr %j137, align 8
  br label %for.cond138

for.cond138:                                      ; preds = %for.inc146, %for.body136
  %87 = load i64, ptr %j137, align 8
  %88 = load i64, ptr %J, align 8
  %cmp139 = icmp ult i64 %87, %88
  br i1 %cmp139, label %for.body141, label %for.end148

for.body141:                                      ; preds = %for.cond138
  %89 = load i64, ptr %i132, align 8
  %90 = mul nsw i64 %89, %11
  %arrayidx142 = getelementptr inbounds i8, ptr %vla3, i64 %90
  %91 = load i64, ptr %j137, align 8
  %arrayidx143 = getelementptr inbounds i8, ptr %arrayidx142, i64 %91
  %92 = load i8, ptr %arrayidx143, align 1
  %conv144 = sext i8 %92 to i32
  %call145 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef signext %conv144)
  br label %for.inc146

for.inc146:                                       ; preds = %for.body141
  %93 = load i64, ptr %j137, align 8
  %inc147 = add i64 %93, 1
  store i64 %inc147, ptr %j137, align 8
  br label %for.cond138, !llvm.loop !17

for.end148:                                       ; preds = %for.cond138
  %call149 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.2)
  br label %for.inc150

for.inc150:                                       ; preds = %for.end148
  %94 = load i64, ptr %i132, align 8
  %inc151 = add i64 %94, 1
  store i64 %inc151, ptr %i132, align 8
  br label %for.cond133, !llvm.loop !18

for.end152:                                       ; preds = %for.cond133
  %call153 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.12)
  call void@llvm.riscv.configEx(i64 4575657221408489476, i64 281474976710656)
  %call154 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.4)
  call void @llvm.riscv.configSt(i64 2, i64 4575657221408423951)
  %call155 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.5)
  call void @llvm.riscv.configLd(i64 4575657221409472769, i64 16)
  %call156 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.6)
  call void @llvm.riscv.mvin(i64 %81, i64 3940718393425920)
  %call157 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.7)
  call void @llvm.riscv.configLd(i64 4575657221409472769, i64 15)
  %call158 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.6)
  call void @llvm.riscv.mvin(i64 %82, i64 4503664051879952)
  %call159 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.7)
  call void @llvm.riscv.configLd(i64 4575657221409472769, i64 15)
  %call160 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.6)
  call void @llvm.riscv.mvin(i64 %84, i64 3940714098458656)
  %call161 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.7)
  call void @llvm.riscv.preload(i64 4503664051879952, i64 3940714098458672)
  %call162 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.8)
  call void @llvm.riscv.computePreloaded(i64 3940718393425920, i64 3940714098458656)
  %call163 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.9)
  call void @llvm.riscv.mvout(i64 %83, i64 3940714098458672)
  %call164 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.10)
  %call165 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.11)
  call void asm sideeffect "fence", ""()
  store i64 0, ptr %i166, align 8
  br label %for.cond167

for.cond167:                                      ; preds = %for.inc184, %for.end152
  %95 = load i64, ptr %i166, align 8
  %96 = load i64, ptr %I, align 8
  %cmp168 = icmp ult i64 %95, %96
  br i1 %cmp168, label %for.body170, label %for.end186

for.body170:                                      ; preds = %for.cond167
  store i64 0, ptr %j171, align 8
  br label %for.cond172

for.cond172:                                      ; preds = %for.inc180, %for.body170
  %97 = load i64, ptr %j171, align 8
  %98 = load i64, ptr %J, align 8
  %cmp173 = icmp ult i64 %97, %98
  br i1 %cmp173, label %for.body175, label %for.end182

for.body175:                                      ; preds = %for.cond172
  %99 = load i64, ptr %i166, align 8
  %100 = mul nsw i64 %99, %11
  %arrayidx176 = getelementptr inbounds i8, ptr %vla3, i64 %100
  %101 = load i64, ptr %j171, align 8
  %arrayidx177 = getelementptr inbounds i8, ptr %arrayidx176, i64 %101
  %102 = load i8, ptr %arrayidx177, align 1
  %conv178 = sext i8 %102 to i32
  %call179 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef signext %conv178)
  br label %for.inc180

for.inc180:                                       ; preds = %for.body175
  %103 = load i64, ptr %j171, align 8
  %inc181 = add i64 %103, 1
  store i64 %inc181, ptr %j171, align 8
  br label %for.cond172, !llvm.loop !19

for.end182:                                       ; preds = %for.cond172
  %call183 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.2)
  br label %for.inc184

for.inc184:                                       ; preds = %for.end182
  %104 = load i64, ptr %i166, align 8
  %inc185 = add i64 %104, 1
  store i64 %inc185, ptr %i166, align 8
  br label %for.cond167, !llvm.loop !20

for.end186:                                       ; preds = %for.cond167
  store i32 0, ptr %retval, align 4
  %105 = load ptr, ptr %saved_stack, align 8
  call void @llvm.stackrestore(ptr %105)
  %106 = load i32, ptr %retval, align 4
  ret i32 %106
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave() #1
declare dso_local signext i32 @rand() #2
declare dso_local signext i32 @printf(ptr noundef, ...) #2
declare void @llvm.stackrestore(ptr) #1
declare void @llvm.riscv.mvin(i64, i64)
declare void @llvm.riscv.mvout(i64, i64)
declare void @llvm.riscv.configLd(i64, i64)
declare void @llvm.riscv.configSt(i64, i64)
declare void @llvm.riscv.configEx(i64, i64)
declare void @llvm.riscv.preload(i64, i64)
declare void @llvm.riscv.computePreloaded(i64, i64)

attributes #0 = { noinline nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+m,+relax,-save-restore" }
attributes #1 = { nocallback nofree nosync nounwind willreturn }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+m,+relax,-save-restore" }

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
