@.str = private unnamed_addr constant [15 x i8] c"print arrayA.\0A\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"%d \00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@.str.3 = private unnamed_addr constant [15 x i8] c"print arrayB.\0A\00", align 1
@.str.4 = private unnamed_addr constant [15 x i8] c"print arrayC.\0A\00", align 1
@.str.5 = private unnamed_addr constant [13 x i8] c"print gold.\0A\00", align 1
@.str.6 = private unnamed_addr constant [19 x i8] c"config_ld arrayA.\0A\00", align 1
@.str.7 = private unnamed_addr constant [14 x i8] c"mvin arrayA.\0A\00", align 1
@.str.8 = private unnamed_addr constant [19 x i8] c"config_ld arrayB.\0A\00", align 1
@.str.9 = private unnamed_addr constant [14 x i8] c"mvin arrayB.\0A\00", align 1
@.str.10 = private unnamed_addr constant [12 x i8] c"config_ex.\0A\00", align 1
@.str.11 = private unnamed_addr constant [12 x i8] c"config_st.\0A\00", align 1
@.str.12 = private unnamed_addr constant [14 x i8] c"mvout arrayC\0A\00", align 1

define dso_local signext i32 @main() {
entry:
  %retval = alloca i32, align 4
  %arrayA = alloca [16 x [16 x i8]], align 1
  %arrayB = alloca [16 x [16 x i8]], align 1
  %arrayC = alloca [16 x [16 x i8]], align 1
  %gold = alloca [16 x [16 x i8]], align 1
  %i = alloca i64, align 8
  %j = alloca i64, align 8
  %i14 = alloca i64, align 8
  %j19 = alloca i64, align 8
  %sum = alloca i32, align 4
  %y = alloca float, align 4
  %x_ = alloca i32, align 4
  %i27 = alloca i64, align 8
  %next = alloca i64, align 8
  %rem33 = alloca i32, align 4
  %result = alloca i32, align 4
  %tmp = alloca i32, align 4
  %tmp67 = alloca i32, align 4
  %y82 = alloca float, align 4
  %x_83 = alloca i32, align 4
  %i88 = alloca i64, align 8
  %next90 = alloca i64, align 8
  %rem100 = alloca i32, align 4
  %result111 = alloca i32, align 4
  %tmp134 = alloca i32, align 4
  %tmp136 = alloca i32, align 4
  %i174 = alloca i64, align 8
  %j179 = alloca i64, align 8
  %i196 = alloca i64, align 8
  %j201 = alloca i64, align 8
  %i218 = alloca i64, align 8
  %j223 = alloca i64, align 8
  %i240 = alloca i64, align 8
  %j245 = alloca i64, align 8
  %A_acc_addr = alloca i32, align 4
  %temp = alloca i64, align 8
  %B_acc_addr = alloca i32, align 4
  %C_acc_addr = alloca i32, align 4
  %addrA = alloca i64, align 8
  %addrB = alloca i64, align 8
  %addrC = alloca i64, align 8
  %i274 = alloca i64, align 8
  %j279 = alloca i64, align 8
  %i296 = alloca i64, align 8
  %j301 = alloca i64, align 8
  store i32 0, ptr %retval, align 4
  store i64 0, ptr %i, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.inc11, %entry
  %0 = load i64, ptr %i, align 8
  %cmp = icmp ult i64 %0, 16
  br i1 %cmp, label %for.body, label %for.end13

for.body:                                         ; preds = %for.cond
  store i64 0, ptr %j, align 8
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %1 = load i64, ptr %j, align 8
  %cmp2 = icmp ult i64 %1, 16
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %call = call signext i32 @rand()
  %rem = srem i32 %call, 16
  %sub = sub nsw i32 %rem, 8
  %conv = trunc i32 %sub to i8
  %2 = load i64, ptr %i, align 8
  %arrayidx = getelementptr inbounds [16 x [16 x i8]], ptr %arrayA, i64 0, i64 %2
  %3 = load i64, ptr %j, align 8
  %arrayidx4 = getelementptr inbounds [16 x i8], ptr %arrayidx, i64 0, i64 %3
  store i8 %conv, ptr %arrayidx4, align 1
  %call5 = call signext i32 @rand()
  %rem6 = srem i32 %call5, 16
  %sub7 = sub nsw i32 %rem6, 8
  %conv8 = trunc i32 %sub7 to i8
  %4 = load i64, ptr %i, align 8
  %arrayidx9 = getelementptr inbounds [16 x [16 x i8]], ptr %arrayB, i64 0, i64 %4
  %5 = load i64, ptr %j, align 8
  %arrayidx10 = getelementptr inbounds [16 x i8], ptr %arrayidx9, i64 0, i64 %5
  store i8 %conv8, ptr %arrayidx10, align 1
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %6 = load i64, ptr %j, align 8
  %inc = add i64 %6, 1
  store i64 %inc, ptr %j, align 8
  br label %for.cond1, !llvm.loop !5

for.end:                                          ; preds = %for.cond1
  br label %for.inc11

for.inc11:                                        ; preds = %for.end
  %7 = load i64, ptr %i, align 8
  %inc12 = add i64 %7, 1
  store i64 %inc12, ptr %i, align 8
  br label %for.cond, !llvm.loop !7

for.end13:                                        ; preds = %for.cond
  store i64 0, ptr %i14, align 8
  br label %for.cond15

for.cond15:                                       ; preds = %for.inc170, %for.end13
  %8 = load i64, ptr %i14, align 8
  %cmp16 = icmp ult i64 %8, 16
  br i1 %cmp16, label %for.body18, label %for.end172

for.body18:                                       ; preds = %for.cond15
  store i64 0, ptr %j19, align 8
  br label %for.cond20

for.cond20:                                       ; preds = %for.inc167, %for.body18
  %9 = load i64, ptr %j19, align 8
  %cmp21 = icmp ult i64 %9, 16
  br i1 %cmp21, label %for.body23, label %for.end169

for.body23:                                       ; preds = %for.cond20
  %10 = load i64, ptr %i14, align 8
  %arrayidx24 = getelementptr inbounds [16 x [16 x i8]], ptr %arrayA, i64 0, i64 %10
  %11 = load i64, ptr %j19, align 8
  %arrayidx25 = getelementptr inbounds [16 x i8], ptr %arrayidx24, i64 0, i64 %11
  %12 = load i8, ptr %arrayidx25, align 1
  %conv26 = sext i8 %12 to i32
  %mul = mul nsw i32 %conv26, 1
  store i32 %mul, ptr %x_, align 4
  %13 = load i32, ptr %x_, align 4
  %conv28 = sext i32 %13 to i64
  store i64 %conv28, ptr %i27, align 8
  %14 = load i32, ptr %x_, align 4
  %cmp29 = icmp slt i32 %14, 0
  br i1 %cmp29, label %cond.true, label %cond.false

cond.true:                                        ; preds = %for.body23
  %15 = load i32, ptr %x_, align 4
  %sub31 = sub nsw i32 %15, 1
  br label %cond.end

cond.false:                                       ; preds = %for.body23
  %16 = load i32, ptr %x_, align 4
  %add = add nsw i32 %16, 1
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %sub31, %cond.true ], [ %add, %cond.false ]
  %conv32 = sext i32 %cond to i64
  store i64 %conv32, ptr %next, align 8
  %17 = load i32, ptr %x_, align 4
  %conv34 = sext i32 %17 to i64
  %18 = load i64, ptr %i27, align 8
  %sub35 = sub nsw i64 %conv34, %18
  %conv36 = trunc i64 %sub35 to i32
  store i32 %conv36, ptr %rem33, align 4
  %19 = load i32, ptr %rem33, align 4
  %cmp37 = icmp slt i32 %19, 0
  br i1 %cmp37, label %cond.true39, label %cond.false41

cond.true39:                                      ; preds = %cond.end
  %20 = load i32, ptr %rem33, align 4
  %sub40 = sub nsw i32 0, %20
  br label %cond.end42

cond.false41:                                     ; preds = %cond.end
  %21 = load i32, ptr %rem33, align 4
  br label %cond.end42

cond.end42:                                       ; preds = %cond.false41, %cond.true39
  %cond43 = phi i32 [ %sub40, %cond.true39 ], [ %21, %cond.false41 ]
  store i32 %cond43, ptr %rem33, align 4
  %22 = load i32, ptr %rem33, align 4
  %conv44 = sitofp i32 %22 to double
  %cmp45 = fcmp olt double %conv44, 5.000000e-01
  br i1 %cmp45, label %cond.true47, label %cond.false48

cond.true47:                                      ; preds = %cond.end42
  %23 = load i64, ptr %i27, align 8
  br label %cond.end63

cond.false48:                                     ; preds = %cond.end42
  %24 = load i32, ptr %rem33, align 4
  %conv49 = sitofp i32 %24 to double
  %cmp50 = fcmp ogt double %conv49, 5.000000e-01
  br i1 %cmp50, label %cond.true52, label %cond.false53

cond.true52:                                      ; preds = %cond.false48
  %25 = load i64, ptr %next, align 8
  br label %cond.end61

cond.false53:                                     ; preds = %cond.false48
  %26 = load i64, ptr %i27, align 8
  %rem54 = srem i64 %26, 2
  %cmp55 = icmp eq i64 %rem54, 0
  br i1 %cmp55, label %cond.true57, label %cond.false58

cond.true57:                                      ; preds = %cond.false53
  %27 = load i64, ptr %i27, align 8
  br label %cond.end59

cond.false58:                                     ; preds = %cond.false53
  %28 = load i64, ptr %next, align 8
  br label %cond.end59

cond.end59:                                       ; preds = %cond.false58, %cond.true57
  %cond60 = phi i64 [ %27, %cond.true57 ], [ %28, %cond.false58 ]
  br label %cond.end61

cond.end61:                                       ; preds = %cond.end59, %cond.true52
  %cond62 = phi i64 [ %25, %cond.true52 ], [ %cond60, %cond.end59 ]
  br label %cond.end63

cond.end63:                                       ; preds = %cond.end61, %cond.true47
  %cond64 = phi i64 [ %23, %cond.true47 ], [ %cond62, %cond.end61 ]
  %conv65 = trunc i64 %cond64 to i32
  store i32 %conv65, ptr %result, align 4
  %29 = load i32, ptr %result, align 4
  store i32 %29, ptr %tmp, align 4
  %30 = load i32, ptr %tmp, align 4
  %conv66 = sitofp i32 %30 to float
  store float %conv66, ptr %y, align 4
  %31 = load float, ptr %y, align 4
  %cmp68 = fcmp ogt float %31, 1.270000e+02
  br i1 %cmp68, label %cond.true70, label %cond.false71

cond.true70:                                      ; preds = %cond.end63
  br label %cond.end80

cond.false71:                                     ; preds = %cond.end63
  %32 = load float, ptr %y, align 4
  %cmp72 = fcmp olt float %32, -1.280000e+02
  br i1 %cmp72, label %cond.true74, label %cond.false75

cond.true74:                                      ; preds = %cond.false71
  br label %cond.end78

cond.false75:                                     ; preds = %cond.false71
  %33 = load float, ptr %y, align 4
  %conv76 = fptosi float %33 to i8
  %conv77 = sext i8 %conv76 to i32
  br label %cond.end78

cond.end78:                                       ; preds = %cond.false75, %cond.true74
  %cond79 = phi i32 [ -128, %cond.true74 ], [ %conv77, %cond.false75 ]
  br label %cond.end80

cond.end80:                                       ; preds = %cond.end78, %cond.true70
  %cond81 = phi i32 [ 127, %cond.true70 ], [ %cond79, %cond.end78 ]
  store i32 %cond81, ptr %tmp67, align 4
  %34 = load i32, ptr %tmp67, align 4
  %35 = load i64, ptr %i14, align 8
  %arrayidx84 = getelementptr inbounds [16 x [16 x i8]], ptr %arrayB, i64 0, i64 %35
  %36 = load i64, ptr %j19, align 8
  %arrayidx85 = getelementptr inbounds [16 x i8], ptr %arrayidx84, i64 0, i64 %36
  %37 = load i8, ptr %arrayidx85, align 1
  %conv86 = sext i8 %37 to i32
  %mul87 = mul nsw i32 %conv86, 1
  store i32 %mul87, ptr %x_83, align 4
  %38 = load i32, ptr %x_83, align 4
  %conv89 = sext i32 %38 to i64
  store i64 %conv89, ptr %i88, align 8
  %39 = load i32, ptr %x_83, align 4
  %cmp91 = icmp slt i32 %39, 0
  br i1 %cmp91, label %cond.true93, label %cond.false95

cond.true93:                                      ; preds = %cond.end80
  %40 = load i32, ptr %x_83, align 4
  %sub94 = sub nsw i32 %40, 1
  br label %cond.end97

cond.false95:                                     ; preds = %cond.end80
  %41 = load i32, ptr %x_83, align 4
  %add96 = add nsw i32 %41, 1
  br label %cond.end97

cond.end97:                                       ; preds = %cond.false95, %cond.true93
  %cond98 = phi i32 [ %sub94, %cond.true93 ], [ %add96, %cond.false95 ]
  %conv99 = sext i32 %cond98 to i64
  store i64 %conv99, ptr %next90, align 8
  %42 = load i32, ptr %x_83, align 4
  %conv101 = sext i32 %42 to i64
  %43 = load i64, ptr %i88, align 8
  %sub102 = sub nsw i64 %conv101, %43
  %conv103 = trunc i64 %sub102 to i32
  store i32 %conv103, ptr %rem100, align 4
  %44 = load i32, ptr %rem100, align 4
  %cmp104 = icmp slt i32 %44, 0
  br i1 %cmp104, label %cond.true106, label %cond.false108

cond.true106:                                     ; preds = %cond.end97
  %45 = load i32, ptr %rem100, align 4
  %sub107 = sub nsw i32 0, %45
  br label %cond.end109

cond.false108:                                    ; preds = %cond.end97
  %46 = load i32, ptr %rem100, align 4
  br label %cond.end109

cond.end109:                                      ; preds = %cond.false108, %cond.true106
  %cond110 = phi i32 [ %sub107, %cond.true106 ], [ %46, %cond.false108 ]
  store i32 %cond110, ptr %rem100, align 4
  %47 = load i32, ptr %rem100, align 4
  %conv112 = sitofp i32 %47 to double
  %cmp113 = fcmp olt double %conv112, 5.000000e-01
  br i1 %cmp113, label %cond.true115, label %cond.false116

cond.true115:                                     ; preds = %cond.end109
  %48 = load i64, ptr %i88, align 8
  br label %cond.end131

cond.false116:                                    ; preds = %cond.end109
  %49 = load i32, ptr %rem100, align 4
  %conv117 = sitofp i32 %49 to double
  %cmp118 = fcmp ogt double %conv117, 5.000000e-01
  br i1 %cmp118, label %cond.true120, label %cond.false121

cond.true120:                                     ; preds = %cond.false116
  %50 = load i64, ptr %next90, align 8
  br label %cond.end129

cond.false121:                                    ; preds = %cond.false116
  %51 = load i64, ptr %i88, align 8
  %rem122 = srem i64 %51, 2
  %cmp123 = icmp eq i64 %rem122, 0
  br i1 %cmp123, label %cond.true125, label %cond.false126

cond.true125:                                     ; preds = %cond.false121
  %52 = load i64, ptr %i88, align 8
  br label %cond.end127

cond.false126:                                    ; preds = %cond.false121
  %53 = load i64, ptr %next90, align 8
  br label %cond.end127

cond.end127:                                      ; preds = %cond.false126, %cond.true125
  %cond128 = phi i64 [ %52, %cond.true125 ], [ %53, %cond.false126 ]
  br label %cond.end129

cond.end129:                                      ; preds = %cond.end127, %cond.true120
  %cond130 = phi i64 [ %50, %cond.true120 ], [ %cond128, %cond.end127 ]
  br label %cond.end131

cond.end131:                                      ; preds = %cond.end129, %cond.true115
  %cond132 = phi i64 [ %48, %cond.true115 ], [ %cond130, %cond.end129 ]
  %conv133 = trunc i64 %cond132 to i32
  store i32 %conv133, ptr %result111, align 4
  %54 = load i32, ptr %result111, align 4
  store i32 %54, ptr %tmp134, align 4
  %55 = load i32, ptr %tmp134, align 4
  %conv135 = sitofp i32 %55 to float
  store float %conv135, ptr %y82, align 4
  %56 = load float, ptr %y82, align 4
  %cmp137 = fcmp ogt float %56, 1.270000e+02
  br i1 %cmp137, label %cond.true139, label %cond.false140

cond.true139:                                     ; preds = %cond.end131
  br label %cond.end149

cond.false140:                                    ; preds = %cond.end131
  %57 = load float, ptr %y82, align 4
  %cmp141 = fcmp olt float %57, -1.280000e+02
  br i1 %cmp141, label %cond.true143, label %cond.false144

cond.true143:                                     ; preds = %cond.false140
  br label %cond.end147

cond.false144:                                    ; preds = %cond.false140
  %58 = load float, ptr %y82, align 4
  %conv145 = fptosi float %58 to i8
  %conv146 = sext i8 %conv145 to i32
  br label %cond.end147

cond.end147:                                      ; preds = %cond.false144, %cond.true143
  %cond148 = phi i32 [ -128, %cond.true143 ], [ %conv146, %cond.false144 ]
  br label %cond.end149

cond.end149:                                      ; preds = %cond.end147, %cond.true139
  %cond150 = phi i32 [ 127, %cond.true139 ], [ %cond148, %cond.end147 ]
  store i32 %cond150, ptr %tmp136, align 4
  %59 = load i32, ptr %tmp136, align 4
  %add151 = add nsw i32 %34, %59
  store i32 %add151, ptr %sum, align 4
  %60 = load i32, ptr %sum, align 4
  %cmp152 = icmp sgt i32 %60, 127
  br i1 %cmp152, label %cond.true154, label %cond.false155

cond.true154:                                     ; preds = %cond.end149
  br label %cond.end162

cond.false155:                                    ; preds = %cond.end149
  %61 = load i32, ptr %sum, align 4
  %cmp156 = icmp slt i32 %61, -128
  br i1 %cmp156, label %cond.true158, label %cond.false159

cond.true158:                                     ; preds = %cond.false155
  br label %cond.end160

cond.false159:                                    ; preds = %cond.false155
  %62 = load i32, ptr %sum, align 4
  br label %cond.end160

cond.end160:                                      ; preds = %cond.false159, %cond.true158
  %cond161 = phi i32 [ -128, %cond.true158 ], [ %62, %cond.false159 ]
  br label %cond.end162

cond.end162:                                      ; preds = %cond.end160, %cond.true154
  %cond163 = phi i32 [ 127, %cond.true154 ], [ %cond161, %cond.end160 ]
  %conv164 = trunc i32 %cond163 to i8
  %63 = load i64, ptr %i14, align 8
  %arrayidx165 = getelementptr inbounds [16 x [16 x i8]], ptr %gold, i64 0, i64 %63
  %64 = load i64, ptr %j19, align 8
  %arrayidx166 = getelementptr inbounds [16 x i8], ptr %arrayidx165, i64 0, i64 %64
  store i8 %conv164, ptr %arrayidx166, align 1
  br label %for.inc167

for.inc167:                                       ; preds = %cond.end162
  %65 = load i64, ptr %j19, align 8
  %inc168 = add i64 %65, 1
  store i64 %inc168, ptr %j19, align 8
  br label %for.cond20, !llvm.loop !8

for.end169:                                       ; preds = %for.cond20
  br label %for.inc170

for.inc170:                                       ; preds = %for.end169
  %66 = load i64, ptr %i14, align 8
  %inc171 = add i64 %66, 1
  store i64 %inc171, ptr %i14, align 8
  br label %for.cond15, !llvm.loop !9

for.end172:                                       ; preds = %for.cond15
  %call173 = call signext i32 (ptr, ...) @printf(ptr noundef @.str)
  store i64 0, ptr %i174, align 8
  br label %for.cond175

for.cond175:                                      ; preds = %for.inc192, %for.end172
  %67 = load i64, ptr %i174, align 8
  %cmp176 = icmp ult i64 %67, 16
  br i1 %cmp176, label %for.body178, label %for.end194

for.body178:                                      ; preds = %for.cond175
  store i64 0, ptr %j179, align 8
  br label %for.cond180

for.cond180:                                      ; preds = %for.inc188, %for.body178
  %68 = load i64, ptr %j179, align 8
  %cmp181 = icmp ult i64 %68, 16
  br i1 %cmp181, label %for.body183, label %for.end190

for.body183:                                      ; preds = %for.cond180
  %69 = load i64, ptr %i174, align 8
  %arrayidx184 = getelementptr inbounds [16 x [16 x i8]], ptr %arrayA, i64 0, i64 %69
  %70 = load i64, ptr %j179, align 8
  %arrayidx185 = getelementptr inbounds [16 x i8], ptr %arrayidx184, i64 0, i64 %70
  %71 = load i8, ptr %arrayidx185, align 1
  %conv186 = sext i8 %71 to i32
  %call187 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef signext %conv186)
  br label %for.inc188

for.inc188:                                       ; preds = %for.body183
  %72 = load i64, ptr %j179, align 8
  %inc189 = add i64 %72, 1
  store i64 %inc189, ptr %j179, align 8
  br label %for.cond180, !llvm.loop !10

for.end190:                                       ; preds = %for.cond180
  %call191 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.2)
  br label %for.inc192

for.inc192:                                       ; preds = %for.end190
  %73 = load i64, ptr %i174, align 8
  %inc193 = add i64 %73, 1
  store i64 %inc193, ptr %i174, align 8
  br label %for.cond175, !llvm.loop !11

for.end194:                                       ; preds = %for.cond175
  %call195 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.3)
  store i64 0, ptr %i196, align 8
  br label %for.cond197

for.cond197:                                      ; preds = %for.inc214, %for.end194
  %74 = load i64, ptr %i196, align 8
  %cmp198 = icmp ult i64 %74, 16
  br i1 %cmp198, label %for.body200, label %for.end216

for.body200:                                      ; preds = %for.cond197
  store i64 0, ptr %j201, align 8
  br label %for.cond202

for.cond202:                                      ; preds = %for.inc210, %for.body200
  %75 = load i64, ptr %j201, align 8
  %cmp203 = icmp ult i64 %75, 16
  br i1 %cmp203, label %for.body205, label %for.end212

for.body205:                                      ; preds = %for.cond202
  %76 = load i64, ptr %j201, align 8
  %arrayidx206 = getelementptr inbounds [16 x [16 x i8]], ptr %arrayB, i64 0, i64 %76
  %77 = load i64, ptr %j201, align 8
  %arrayidx207 = getelementptr inbounds [16 x i8], ptr %arrayidx206, i64 0, i64 %77
  %78 = load i8, ptr %arrayidx207, align 1
  %conv208 = sext i8 %78 to i32
  %call209 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef signext %conv208)
  br label %for.inc210

for.inc210:                                       ; preds = %for.body205
  %79 = load i64, ptr %j201, align 8
  %inc211 = add i64 %79, 1
  store i64 %inc211, ptr %j201, align 8
  br label %for.cond202, !llvm.loop !12

for.end212:                                       ; preds = %for.cond202
  %call213 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.2)
  br label %for.inc214

for.inc214:                                       ; preds = %for.end212
  %80 = load i64, ptr %i196, align 8
  %inc215 = add i64 %80, 1
  store i64 %inc215, ptr %i196, align 8
  br label %for.cond197, !llvm.loop !13

for.end216:                                       ; preds = %for.cond197
  %call217 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.4)
  store i64 0, ptr %i218, align 8
  br label %for.cond219

for.cond219:                                      ; preds = %for.inc236, %for.end216
  %81 = load i64, ptr %i218, align 8
  %cmp220 = icmp ult i64 %81, 16
  br i1 %cmp220, label %for.body222, label %for.end238

for.body222:                                      ; preds = %for.cond219
  store i64 0, ptr %j223, align 8
  br label %for.cond224

for.cond224:                                      ; preds = %for.inc232, %for.body222
  %82 = load i64, ptr %j223, align 8
  %cmp225 = icmp ult i64 %82, 16
  br i1 %cmp225, label %for.body227, label %for.end234

for.body227:                                      ; preds = %for.cond224
  %83 = load i64, ptr %i218, align 8
  %arrayidx228 = getelementptr inbounds [16 x [16 x i8]], ptr %arrayC, i64 0, i64 %83
  %84 = load i64, ptr %j223, align 8
  %arrayidx229 = getelementptr inbounds [16 x i8], ptr %arrayidx228, i64 0, i64 %84
  %85 = load i8, ptr %arrayidx229, align 1
  %conv230 = sext i8 %85 to i32
  %call231 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef signext %conv230)
  br label %for.inc232

for.inc232:                                       ; preds = %for.body227
  %86 = load i64, ptr %j223, align 8
  %inc233 = add i64 %86, 1
  store i64 %inc233, ptr %j223, align 8
  br label %for.cond224, !llvm.loop !14

for.end234:                                       ; preds = %for.cond224
  %call235 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.2)
  br label %for.inc236

for.inc236:                                       ; preds = %for.end234
  %87 = load i64, ptr %i218, align 8
  %inc237 = add i64 %87, 1
  store i64 %inc237, ptr %i218, align 8
  br label %for.cond219, !llvm.loop !15

for.end238:                                       ; preds = %for.cond219
  %call239 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.5)
  store i64 0, ptr %i240, align 8
  br label %for.cond241

for.cond241:                                      ; preds = %for.inc258, %for.end238
  %88 = load i64, ptr %i240, align 8
  %cmp242 = icmp ult i64 %88, 16
  br i1 %cmp242, label %for.body244, label %for.end260

for.body244:                                      ; preds = %for.cond241
  store i64 0, ptr %j245, align 8
  br label %for.cond246

for.cond246:                                      ; preds = %for.inc254, %for.body244
  %89 = load i64, ptr %j245, align 8
  %cmp247 = icmp ult i64 %89, 16
  br i1 %cmp247, label %for.body249, label %for.end256

for.body249:                                      ; preds = %for.cond246
  %90 = load i64, ptr %i240, align 8
  %arrayidx250 = getelementptr inbounds [16 x [16 x i8]], ptr %gold, i64 0, i64 %90
  %91 = load i64, ptr %j245, align 8
  %arrayidx251 = getelementptr inbounds [16 x i8], ptr %arrayidx250, i64 0, i64 %91
  %92 = load i8, ptr %arrayidx251, align 1
  %conv252 = sext i8 %92 to i32
  %call253 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef signext %conv252)
  br label %for.inc254

for.inc254:                                       ; preds = %for.body249
  %93 = load i64, ptr %j245, align 8
  %inc255 = add i64 %93, 1
  store i64 %inc255, ptr %j245, align 8
  br label %for.cond246, !llvm.loop !16

for.end256:                                       ; preds = %for.cond246
  %call257 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.2)
  br label %for.inc258

for.inc258:                                       ; preds = %for.end256
  %94 = load i64, ptr %i240, align 8
  %inc259 = add i64 %94, 1
  store i64 %inc259, ptr %i240, align 8
  br label %for.cond241, !llvm.loop !17

for.end260:                                       ; preds = %for.cond241
  store i32 -2147483648, ptr %A_acc_addr, align 4
  %95 = load i32, ptr %A_acc_addr, align 4
  %conv261 = zext i32 %95 to i64
  store i64 %conv261, ptr %temp, align 8
  store i32 -1073741824, ptr %B_acc_addr, align 4
  %96 = load i32, ptr %B_acc_addr, align 4
  %conv262 = zext i32 %96 to i64
  store i64 %conv262, ptr %temp, align 8
  store i32 -2147483648, ptr %C_acc_addr, align 4
  %97 = load i32, ptr %C_acc_addr, align 4
  %conv263 = zext i32 %97 to i64
  store i64 %conv263, ptr %temp, align 8
  %arraydecay = getelementptr inbounds [16 x [16 x i8]], ptr %arrayA, i64 0, i64 0
  %98 = ptrtoint ptr %arraydecay to i64
  store i64 %98, ptr %addrA, align 8
  %arraydecay264 = getelementptr inbounds [16 x [16 x i8]], ptr %arrayB, i64 0, i64 0
  %99 = ptrtoint ptr %arraydecay264 to i64
  store i64 %99, ptr %addrB, align 8
  %arraydecay265 = getelementptr inbounds [16 x [16 x i8]], ptr %arrayC, i64 0, i64 0
  %100 = ptrtoint ptr %arraydecay265 to i64
  store i64 %100, ptr %addrC, align 8
  call void @llvm.riscv.configLd(i64 4575657221409472773, i64 16)
  %call266 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.6)
  call void @llvm.riscv.mvin(i64 %98, i64 4503670494330880)
  %call267 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.7)
  call void @llvm.riscv.configLd(i64 4575657221409472773, i64 16)
  %call268 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.8)
  call void @llvm.riscv.mvin(i64 %99, i64 4503671568072704)
  %call269 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.9)
  call void @llvm.riscv.configEx(i64 4575657221408489472, i64 281474976710656)
  %call270 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.10)
  call void @llvm.riscv.configSt(i64 2, i64 4575657221408423952)
  %call271 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.11)
  call void @llvm.riscv.mvout(i64 %100, i64 4503670494330880)
  %call272 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.12)
  %call273 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.5)
  store i64 0, ptr %i274, align 8
  br label %for.cond275

for.cond275:                                      ; preds = %for.inc292, %for.end260
  %101 = load i64, ptr %i274, align 8
  %cmp276 = icmp ult i64 %101, 16
  br i1 %cmp276, label %for.body278, label %for.end294

for.body278:                                      ; preds = %for.cond275
  store i64 0, ptr %j279, align 8
  br label %for.cond280

for.cond280:                                      ; preds = %for.inc288, %for.body278
  %102 = load i64, ptr %j279, align 8
  %cmp281 = icmp ult i64 %102, 16
  br i1 %cmp281, label %for.body283, label %for.end290

for.body283:                                      ; preds = %for.cond280
  %103 = load i64, ptr %i274, align 8
  %arrayidx284 = getelementptr inbounds [16 x [16 x i8]], ptr %gold, i64 0, i64 %103
  %104 = load i64, ptr %j279, align 8
  %arrayidx285 = getelementptr inbounds [16 x i8], ptr %arrayidx284, i64 0, i64 %104
  %105 = load i8, ptr %arrayidx285, align 1
  %conv286 = sext i8 %105 to i32
  %call287 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef signext %conv286)
  br label %for.inc288

for.inc288:                                       ; preds = %for.body283
  %106 = load i64, ptr %j279, align 8
  %inc289 = add i64 %106, 1
  store i64 %inc289, ptr %j279, align 8
  br label %for.cond280, !llvm.loop !18

for.end290:                                       ; preds = %for.cond280
  %call291 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.2)
  br label %for.inc292

for.inc292:                                       ; preds = %for.end290
  %107 = load i64, ptr %i274, align 8
  %inc293 = add i64 %107, 1
  store i64 %inc293, ptr %i274, align 8
  br label %for.cond275, !llvm.loop !19

for.end294:                                       ; preds = %for.cond275
  %call295 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.4)
  store i64 0, ptr %i296, align 8
  br label %for.cond297

for.cond297:                                      ; preds = %for.inc314, %for.end294
  %108 = load i64, ptr %i296, align 8
  %cmp298 = icmp ult i64 %108, 16
  br i1 %cmp298, label %for.body300, label %for.end316

for.body300:                                      ; preds = %for.cond297
  store i64 0, ptr %j301, align 8
  br label %for.cond302

for.cond302:                                      ; preds = %for.inc310, %for.body300
  %109 = load i64, ptr %j301, align 8
  %cmp303 = icmp ult i64 %109, 16
  br i1 %cmp303, label %for.body305, label %for.end312

for.body305:                                      ; preds = %for.cond302
  %110 = load i64, ptr %i296, align 8
  %arrayidx306 = getelementptr inbounds [16 x [16 x i8]], ptr %arrayC, i64 0, i64 %110
  %111 = load i64, ptr %j301, align 8
  %arrayidx307 = getelementptr inbounds [16 x i8], ptr %arrayidx306, i64 0, i64 %111
  %112 = load i8, ptr %arrayidx307, align 1
  %conv308 = sext i8 %112 to i32
  %call309 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef signext %conv308)
  br label %for.inc310

for.inc310:                                       ; preds = %for.body305
  %113 = load i64, ptr %j301, align 8
  %inc311 = add i64 %113, 1
  store i64 %inc311, ptr %j301, align 8
  br label %for.cond302, !llvm.loop !20

for.end312:                                       ; preds = %for.cond302
  %call313 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.2)
  br label %for.inc314

for.inc314:                                       ; preds = %for.end312
  %114 = load i64, ptr %i296, align 8
  %inc315 = add i64 %114, 1
  store i64 %inc315, ptr %i296, align 8
  br label %for.cond297, !llvm.loop !21

for.end316:                                       ; preds = %for.cond297
  ret i32 0
}

declare dso_local signext i32 @rand() 
declare dso_local signext i32 @printf(ptr noundef, ...)
declare void @llvm.riscv.mvin(i64, i64)
declare void @llvm.riscv.mvout(i64, i64)
declare void @llvm.riscv.flush(i64, i64)
declare void @llvm.riscv.configLd(i64, i64)
declare void @llvm.riscv.configSt(i64, i64)
declare void @llvm.riscv.configEx(i64, i64)


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
