; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

declare ptr @malloc(i64)

define void @matmul_kernel(i64 %0, ptr %1, i64 %2, ptr %3, i64 %4, ptr %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10, i32 %11, i32 %12, i32 %13, i32 %14, i32 %15, i32 %16, i32 %17) {
  %19 = insertvalue { i64, ptr } poison, i64 %4, 0
  %20 = insertvalue { i64, ptr } %19, ptr %5, 1
  %21 = insertvalue { i64, ptr } poison, i64 %2, 0
  %22 = insertvalue { i64, ptr } %21, ptr %3, 1
  %23 = insertvalue { i64, ptr } poison, i64 %0, 0
  %24 = insertvalue { i64, ptr } %23, ptr %1, 1
  %25 = call ptr @malloc(i64 8256)
  %26 = ptrtoint ptr %25 to i64
  %27 = add i64 %26, 63
  %28 = urem i64 %27, 64
  %29 = sub i64 %27, %28
  %30 = inttoptr i64 %29 to ptr
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %25, 0
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, ptr %30, 1
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 0, 2
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 32, 3, 0
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, i64 64, 3, 1
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, i64 64, 4, 0
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, i64 1, 4, 1
  br label %38

38:                                               ; preds = %61, %18
  %39 = phi i64 [ %62, %61 ], [ 0, %18 ]
  %40 = icmp slt i64 %39, 32
  br i1 %40, label %41, label %63

41:                                               ; preds = %38
  br label %42

42:                                               ; preds = %45, %41
  %43 = phi i64 [ %50, %45 ], [ 0, %41 ]
  %44 = icmp slt i64 %43, 61
  br i1 %44, label %45, label %51

45:                                               ; preds = %42
  %46 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 1
  %47 = mul i64 %39, 64
  %48 = add i64 %47, %43
  %49 = getelementptr float, ptr %46, i64 %48
  store <4 x float> zeroinitializer, ptr %49, align 4
  %50 = add i64 %43, 4
  br label %42

51:                                               ; preds = %42
  br label %52

52:                                               ; preds = %55, %51
  %53 = phi i64 [ %60, %55 ], [ 64, %51 ]
  %54 = icmp slt i64 %53, 64
  br i1 %54, label %55, label %61

55:                                               ; preds = %52
  %56 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 1
  %57 = mul nuw nsw i64 %39, 64
  %58 = add nuw nsw i64 %57, %53
  %59 = getelementptr inbounds nuw float, ptr %56, i64 %58
  store float 0.000000e+00, ptr %59, align 4
  %60 = add i64 %53, 1
  br label %52

61:                                               ; preds = %52
  %62 = add i64 %39, 1
  br label %38

63:                                               ; preds = %38
  %64 = add i32 %6, 31
  %65 = sdiv i32 %64, 32
  %66 = add i32 %7, 63
  %67 = sdiv i32 %66, 64
  %68 = mul i32 %67, 8
  %69 = sdiv i32 %15, %68
  %70 = mul i32 %69, 8
  %71 = sub i32 %65, %70
  %72 = call i32 @llvm.smin.i32(i32 %71, i32 8)
  %73 = srem i32 %15, %72
  %74 = add i32 %70, %73
  %75 = srem i32 %15, %68
  %76 = sdiv i32 %75, %72
  %77 = mul i32 %74, 32
  %78 = sext i32 %77 to i64
  %79 = mul i32 %76, 64
  %80 = sext i32 %79 to i64
  %81 = sext i32 %6 to i64
  %82 = sext i32 %9 to i64
  %83 = mul i64 %78, %82
  %84 = mul i64 %81, %82
  %85 = sext i32 %10 to i64
  %86 = sext i32 %7 to i64
  %87 = add i32 %8, 15
  %88 = sdiv i32 %87, 16
  %89 = mul i32 %10, 16
  %90 = sext i32 %89 to i64
  %91 = call ptr @malloc(i64 8256)
  %92 = ptrtoint ptr %91 to i64
  %93 = add i64 %92, 63
  %94 = urem i64 %93, 64
  %95 = sub i64 %93, %94
  %96 = inttoptr i64 %95 to ptr
  %97 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %91, 0
  %98 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %97, ptr %96, 1
  %99 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %98, i64 0, 2
  %100 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %99, i64 32, 3, 0
  %101 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %100, i64 64, 3, 1
  %102 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %101, i64 64, 4, 0
  %103 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %102, i64 1, 4, 1
  %104 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 3, 0
  %105 = mul i64 1, %104
  %106 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 3, 1
  %107 = mul i64 %105, %106
  %108 = mul i64 %107, 4
  %109 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 1
  %110 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 2
  %111 = getelementptr float, ptr %109, i64 %110
  %112 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %103, 1
  %113 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %103, 2
  %114 = getelementptr float, ptr %112, i64 %113
  call void @llvm.memcpy.p0.p0.i64(ptr %114, ptr %111, i64 %108, i1 false)
  br label %115

115:                                              ; preds = %550, %63
  %116 = phi i32 [ %553, %550 ], [ 0, %63 ]
  %117 = phi i64 [ %551, %550 ], [ %83, %63 ]
  %118 = phi i64 [ %552, %550 ], [ 0, %63 ]
  %119 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %119, %550 ], [ %103, %63 ]
  %120 = icmp slt i32 %116, %88
  br i1 %120, label %121, label %554

121:                                              ; preds = %115
  %122 = add i64 %118, %80
  %123 = srem i64 %122, %86
  %124 = sub i64 %122, %123
  %125 = add i64 %123, 64
  %126 = call i64 @llvm.smin.i64(i64 %125, i64 %86)
  %127 = sub i64 %126, %123
  %128 = extractvalue { i64, ptr } %22, 1
  %129 = load ptr, ptr %128, align 8
  %130 = getelementptr ptr, ptr %128, i32 1
  %131 = load ptr, ptr %130, align 8
  %132 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %129, 0
  %133 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %132, ptr %131, 1
  %134 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %133, i64 %122, 2
  %135 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %134, i64 16, 3, 0
  %136 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %135, i64 %85, 4, 0
  %137 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %136, i64 %127, 3, 1
  %138 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %137, i64 1, 4, 1
  %139 = sub i64 64, %127
  %140 = extractvalue { i64, ptr } %22, 1
  %141 = load ptr, ptr %140, align 8
  %142 = getelementptr ptr, ptr %140, i32 1
  %143 = load ptr, ptr %142, align 8
  %144 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %141, 0
  %145 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %144, ptr %143, 1
  %146 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %145, i64 %124, 2
  %147 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %146, i64 16, 3, 0
  %148 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %147, i64 %85, 4, 0
  %149 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %148, i64 %139, 3, 1
  %150 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %149, i64 1, 4, 1
  %151 = srem i64 %117, %82
  %152 = add i64 %84, %151
  %153 = sub i64 %152, %117
  %154 = sdiv i64 %153, %82
  %155 = call i64 @llvm.smin.i64(i64 %154, i64 32)
  %156 = extractvalue { i64, ptr } %24, 1
  %157 = load ptr, ptr %156, align 8
  %158 = getelementptr ptr, ptr %156, i32 1
  %159 = load ptr, ptr %158, align 8
  %160 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %157, 0
  %161 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %160, ptr %159, 1
  %162 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %161, i64 %117, 2
  %163 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %162, i64 %155, 3, 0
  %164 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %163, i64 %82, 4, 0
  %165 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %164, i64 16, 3, 1
  %166 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %165, i64 1, 4, 1
  %167 = sub i64 32, %155
  %168 = extractvalue { i64, ptr } %24, 1
  %169 = load ptr, ptr %168, align 8
  %170 = getelementptr ptr, ptr %168, i32 1
  %171 = load ptr, ptr %170, align 8
  %172 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %169, 0
  %173 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %172, ptr %171, 1
  %174 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %173, i64 %151, 2
  %175 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %174, i64 %167, 3, 0
  %176 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %175, i64 %82, 4, 0
  %177 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %176, i64 16, 3, 1
  %178 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %177, i64 1, 4, 1
  %179 = mul i32 %116, 16
  %180 = sub i32 %8, %179
  %181 = sext i32 %180 to i64
  %182 = call i64 @llvm.smin.i64(i64 %181, i64 16)
  %183 = call i64 @llvm.smax.i64(i64 %182, i64 0)
  %184 = call ptr @malloc(i64 2048)
  %185 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %184, 0
  %186 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %185, ptr %184, 1
  %187 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %186, i64 0, 2
  %188 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %187, i64 32, 3, 0
  %189 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %188, i64 16, 3, 1
  %190 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %189, i64 16, 4, 0
  %191 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %190, i64 1, 4, 1
  %192 = icmp slt i64 %183, 16
  br i1 %192, label %193, label %220

193:                                              ; preds = %121
  br label %194

194:                                              ; preds = %217, %193
  %195 = phi i64 [ %218, %217 ], [ 0, %193 ]
  %196 = icmp slt i64 %195, 32
  br i1 %196, label %197, label %219

197:                                              ; preds = %194
  br label %198

198:                                              ; preds = %201, %197
  %199 = phi i64 [ %206, %201 ], [ 0, %197 ]
  %200 = icmp slt i64 %199, 13
  br i1 %200, label %201, label %207

201:                                              ; preds = %198
  %202 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %191, 1
  %203 = mul i64 %195, 16
  %204 = add i64 %203, %199
  %205 = getelementptr float, ptr %202, i64 %204
  store <4 x float> zeroinitializer, ptr %205, align 4
  %206 = add i64 %199, 4
  br label %198

207:                                              ; preds = %198
  br label %208

208:                                              ; preds = %211, %207
  %209 = phi i64 [ %216, %211 ], [ 16, %207 ]
  %210 = icmp slt i64 %209, 16
  br i1 %210, label %211, label %217

211:                                              ; preds = %208
  %212 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %191, 1
  %213 = mul nuw nsw i64 %195, 16
  %214 = add nuw nsw i64 %213, %209
  %215 = getelementptr inbounds nuw float, ptr %212, i64 %214
  store float 0.000000e+00, ptr %215, align 4
  %216 = add i64 %209, 1
  br label %208

217:                                              ; preds = %208
  %218 = add i64 %195, 1
  br label %194

219:                                              ; preds = %194
  br label %220

220:                                              ; preds = %219, %121
  %221 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %166, 0
  %222 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %166, 1
  %223 = insertvalue { ptr, ptr, i64 } poison, ptr %221, 0
  %224 = insertvalue { ptr, ptr, i64 } %223, ptr %222, 1
  %225 = insertvalue { ptr, ptr, i64 } %224, i64 0, 2
  %226 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %166, 2
  %227 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %166, 3, 0
  %228 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %166, 3, 1
  %229 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %166, 4, 0
  %230 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %166, 4, 1
  %231 = extractvalue { ptr, ptr, i64 } %225, 0
  %232 = extractvalue { ptr, ptr, i64 } %225, 1
  %233 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %231, 0
  %234 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %233, ptr %232, 1
  %235 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %234, i64 %226, 2
  %236 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %235, i64 %155, 3, 0
  %237 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %236, i64 %229, 4, 0
  %238 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %237, i64 %183, 3, 1
  %239 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %238, i64 %230, 4, 1
  %240 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %178, 0
  %241 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %178, 1
  %242 = insertvalue { ptr, ptr, i64 } poison, ptr %240, 0
  %243 = insertvalue { ptr, ptr, i64 } %242, ptr %241, 1
  %244 = insertvalue { ptr, ptr, i64 } %243, i64 0, 2
  %245 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %178, 2
  %246 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %178, 3, 0
  %247 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %178, 3, 1
  %248 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %178, 4, 0
  %249 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %178, 4, 1
  %250 = extractvalue { ptr, ptr, i64 } %244, 0
  %251 = extractvalue { ptr, ptr, i64 } %244, 1
  %252 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %250, 0
  %253 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %252, ptr %251, 1
  %254 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %253, i64 %245, 2
  %255 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %254, i64 %167, 3, 0
  %256 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %255, i64 %248, 4, 0
  %257 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %256, i64 %183, 3, 1
  %258 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %257, i64 %249, 4, 1
  %259 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %191, 0
  %260 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %191, 1
  %261 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %259, 0
  %262 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %261, ptr %260, 1
  %263 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %262, i64 0, 2
  %264 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %263, i64 %155, 3, 0
  %265 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %264, i64 16, 4, 0
  %266 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %265, i64 %183, 3, 1
  %267 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %266, i64 1, 4, 1
  %268 = mul nsw i64 %155, 16
  %269 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %191, 0
  %270 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %191, 1
  %271 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %269, 0
  %272 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %271, ptr %270, 1
  %273 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %272, i64 %268, 2
  %274 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %273, i64 %167, 3, 0
  %275 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %274, i64 16, 4, 0
  %276 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %275, i64 %183, 3, 1
  %277 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %276, i64 1, 4, 1
  %278 = call ptr @llvm.stacksave.p0()
  %279 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %239, ptr %279, align 8
  %280 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %279, 1
  %281 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %267, ptr %281, align 8
  %282 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %281, 1
  %283 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %280, ptr %283, align 8
  %284 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %282, ptr %284, align 8
  call void @memrefCopy(i64 4, ptr %283, ptr %284)
  call void @llvm.stackrestore.p0(ptr %278)
  %285 = call ptr @llvm.stacksave.p0()
  %286 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %258, ptr %286, align 8
  %287 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %286, 1
  %288 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %277, ptr %288, align 8
  %289 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %288, 1
  %290 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %287, ptr %290, align 8
  %291 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %289, ptr %291, align 8
  call void @memrefCopy(i64 4, ptr %290, ptr %291)
  call void @llvm.stackrestore.p0(ptr %285)
  %292 = call ptr @malloc(i64 4096)
  %293 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %292, 0
  %294 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %293, ptr %292, 1
  %295 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %294, i64 0, 2
  %296 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %295, i64 16, 3, 0
  %297 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %296, i64 64, 3, 1
  %298 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %297, i64 64, 4, 0
  %299 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %298, i64 1, 4, 1
  br i1 %192, label %300, label %327

300:                                              ; preds = %220
  br label %301

301:                                              ; preds = %324, %300
  %302 = phi i64 [ %325, %324 ], [ 0, %300 ]
  %303 = icmp slt i64 %302, 16
  br i1 %303, label %304, label %326

304:                                              ; preds = %301
  br label %305

305:                                              ; preds = %308, %304
  %306 = phi i64 [ %313, %308 ], [ 0, %304 ]
  %307 = icmp slt i64 %306, 61
  br i1 %307, label %308, label %314

308:                                              ; preds = %305
  %309 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %299, 1
  %310 = mul i64 %302, 64
  %311 = add i64 %310, %306
  %312 = getelementptr float, ptr %309, i64 %311
  store <4 x float> zeroinitializer, ptr %312, align 4
  %313 = add i64 %306, 4
  br label %305

314:                                              ; preds = %305
  br label %315

315:                                              ; preds = %318, %314
  %316 = phi i64 [ %323, %318 ], [ 64, %314 ]
  %317 = icmp slt i64 %316, 64
  br i1 %317, label %318, label %324

318:                                              ; preds = %315
  %319 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %299, 1
  %320 = mul nuw nsw i64 %302, 64
  %321 = add nuw nsw i64 %320, %316
  %322 = getelementptr inbounds nuw float, ptr %319, i64 %321
  store float 0.000000e+00, ptr %322, align 4
  %323 = add i64 %316, 1
  br label %315

324:                                              ; preds = %315
  %325 = add i64 %302, 1
  br label %301

326:                                              ; preds = %301
  br label %327

327:                                              ; preds = %326, %220
  %328 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %138, 0
  %329 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %138, 1
  %330 = insertvalue { ptr, ptr, i64 } poison, ptr %328, 0
  %331 = insertvalue { ptr, ptr, i64 } %330, ptr %329, 1
  %332 = insertvalue { ptr, ptr, i64 } %331, i64 0, 2
  %333 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %138, 2
  %334 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %138, 3, 0
  %335 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %138, 3, 1
  %336 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %138, 4, 0
  %337 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %138, 4, 1
  %338 = extractvalue { ptr, ptr, i64 } %332, 0
  %339 = extractvalue { ptr, ptr, i64 } %332, 1
  %340 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %338, 0
  %341 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %340, ptr %339, 1
  %342 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %341, i64 %333, 2
  %343 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %342, i64 %183, 3, 0
  %344 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %343, i64 %336, 4, 0
  %345 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %344, i64 %127, 3, 1
  %346 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %345, i64 %337, 4, 1
  %347 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %150, 0
  %348 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %150, 1
  %349 = insertvalue { ptr, ptr, i64 } poison, ptr %347, 0
  %350 = insertvalue { ptr, ptr, i64 } %349, ptr %348, 1
  %351 = insertvalue { ptr, ptr, i64 } %350, i64 0, 2
  %352 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %150, 2
  %353 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %150, 3, 0
  %354 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %150, 3, 1
  %355 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %150, 4, 0
  %356 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %150, 4, 1
  %357 = extractvalue { ptr, ptr, i64 } %351, 0
  %358 = extractvalue { ptr, ptr, i64 } %351, 1
  %359 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %357, 0
  %360 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %359, ptr %358, 1
  %361 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %360, i64 %352, 2
  %362 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %361, i64 %183, 3, 0
  %363 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %362, i64 %355, 4, 0
  %364 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %363, i64 %139, 3, 1
  %365 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %364, i64 %356, 4, 1
  %366 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %299, 0
  %367 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %299, 1
  %368 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %366, 0
  %369 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %368, ptr %367, 1
  %370 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %369, i64 0, 2
  %371 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %370, i64 %183, 3, 0
  %372 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %371, i64 64, 4, 0
  %373 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %372, i64 %127, 3, 1
  %374 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %373, i64 1, 4, 1
  %375 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %299, 0
  %376 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %299, 1
  %377 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %375, 0
  %378 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %377, ptr %376, 1
  %379 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %378, i64 %127, 2
  %380 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %379, i64 %183, 3, 0
  %381 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %380, i64 64, 4, 0
  %382 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %381, i64 %139, 3, 1
  %383 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %382, i64 1, 4, 1
  %384 = call ptr @llvm.stacksave.p0()
  %385 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %346, ptr %385, align 8
  %386 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %385, 1
  %387 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %374, ptr %387, align 8
  %388 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %387, 1
  %389 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %386, ptr %389, align 8
  %390 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %388, ptr %390, align 8
  call void @memrefCopy(i64 4, ptr %389, ptr %390)
  call void @llvm.stackrestore.p0(ptr %384)
  %391 = call ptr @llvm.stacksave.p0()
  %392 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %365, ptr %392, align 8
  %393 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %392, 1
  %394 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %383, ptr %394, align 8
  %395 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %394, 1
  %396 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %393, ptr %396, align 8
  %397 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %395, ptr %397, align 8
  call void @memrefCopy(i64 4, ptr %396, ptr %397)
  call void @llvm.stackrestore.p0(ptr %391)
  %398 = call ptr @malloc(i64 8256)
  %399 = ptrtoint ptr %398 to i64
  %400 = add i64 %399, 63
  %401 = urem i64 %400, 64
  %402 = sub i64 %400, %401
  %403 = inttoptr i64 %402 to ptr
  %404 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %398, 0
  %405 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %404, ptr %403, 1
  %406 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %405, i64 0, 2
  %407 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %406, i64 32, 3, 0
  %408 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %407, i64 64, 3, 1
  %409 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %408, i64 64, 4, 0
  %410 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %409, i64 1, 4, 1
  %411 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 3, 0
  %412 = mul i64 1, %411
  %413 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 3, 1
  %414 = mul i64 %412, %413
  %415 = mul i64 %414, 4
  %416 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 1
  %417 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 2
  %418 = getelementptr float, ptr %416, i64 %417
  %419 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %410, 1
  %420 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %410, 2
  %421 = getelementptr float, ptr %419, i64 %420
  call void @llvm.memcpy.p0.p0.i64(ptr %421, ptr %418, i64 %415, i1 false)
  br label %422

422:                                              ; preds = %460, %327
  %423 = phi i64 [ %461, %460 ], [ 0, %327 ]
  %424 = icmp slt i64 %423, 61
  br i1 %424, label %425, label %462

425:                                              ; preds = %422
  br label %426

426:                                              ; preds = %454, %425
  %427 = phi i64 [ %459, %454 ], [ 0, %425 ]
  %428 = icmp slt i64 %427, 32
  br i1 %428, label %429, label %460

429:                                              ; preds = %426
  %430 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %410, 1
  %431 = mul i64 %427, 64
  %432 = add i64 %431, %423
  %433 = getelementptr float, ptr %430, i64 %432
  %434 = load <4 x float>, ptr %433, align 4
  br label %435

435:                                              ; preds = %439, %429
  %436 = phi i64 [ %453, %439 ], [ 0, %429 ]
  %437 = phi <4 x float> [ %452, %439 ], [ %434, %429 ]
  %438 = icmp slt i64 %436, 16
  br i1 %438, label %439, label %454

439:                                              ; preds = %435
  %440 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %191, 1
  %441 = mul nuw nsw i64 %427, 16
  %442 = add nuw nsw i64 %441, %436
  %443 = getelementptr inbounds nuw float, ptr %440, i64 %442
  %444 = load float, ptr %443, align 4
  %445 = insertelement <4 x float> poison, float %444, i32 0
  %446 = shufflevector <4 x float> %445, <4 x float> poison, <4 x i32> zeroinitializer
  %447 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %299, 1
  %448 = mul i64 %436, 64
  %449 = add i64 %448, %423
  %450 = getelementptr float, ptr %447, i64 %449
  %451 = load <4 x float>, ptr %450, align 4
  %452 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %446, <4 x float> %451, <4 x float> %437)
  %453 = add i64 %436, 1
  br label %435

454:                                              ; preds = %435
  %455 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %410, 1
  %456 = mul i64 %427, 64
  %457 = add i64 %456, %423
  %458 = getelementptr float, ptr %455, i64 %457
  store <4 x float> %437, ptr %458, align 4
  %459 = add i64 %427, 1
  br label %426

460:                                              ; preds = %426
  %461 = add i64 %423, 4
  br label %422

462:                                              ; preds = %422
  br label %463

463:                                              ; preds = %500, %462
  %464 = phi i64 [ %501, %500 ], [ 64, %462 ]
  %465 = icmp slt i64 %464, 64
  br i1 %465, label %466, label %502

466:                                              ; preds = %463
  br label %467

467:                                              ; preds = %494, %466
  %468 = phi i64 [ %499, %494 ], [ 0, %466 ]
  %469 = icmp slt i64 %468, 32
  br i1 %469, label %470, label %500

470:                                              ; preds = %467
  %471 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %410, 1
  %472 = mul nuw nsw i64 %468, 64
  %473 = add nuw nsw i64 %472, %464
  %474 = getelementptr inbounds nuw float, ptr %471, i64 %473
  %475 = load float, ptr %474, align 4
  br label %476

476:                                              ; preds = %480, %470
  %477 = phi i64 [ %493, %480 ], [ 0, %470 ]
  %478 = phi float [ %492, %480 ], [ %475, %470 ]
  %479 = icmp slt i64 %477, 16
  br i1 %479, label %480, label %494

480:                                              ; preds = %476
  %481 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %191, 1
  %482 = mul nuw nsw i64 %468, 16
  %483 = add nuw nsw i64 %482, %477
  %484 = getelementptr inbounds nuw float, ptr %481, i64 %483
  %485 = load float, ptr %484, align 4
  %486 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %299, 1
  %487 = mul nuw nsw i64 %477, 64
  %488 = add nuw nsw i64 %487, %464
  %489 = getelementptr inbounds nuw float, ptr %486, i64 %488
  %490 = load float, ptr %489, align 4
  %491 = fmul float %485, %490
  %492 = fadd float %491, %478
  %493 = add i64 %477, 1
  br label %476

494:                                              ; preds = %476
  %495 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %410, 1
  %496 = mul nuw nsw i64 %468, 64
  %497 = add nuw nsw i64 %496, %464
  %498 = getelementptr inbounds nuw float, ptr %495, i64 %497
  store float %478, ptr %498, align 4
  %499 = add i64 %468, 1
  br label %467

500:                                              ; preds = %467
  %501 = add i64 %464, 1
  br label %463

502:                                              ; preds = %463
  br label %503

503:                                              ; preds = %548, %502
  %504 = phi i64 [ %549, %548 ], [ 0, %502 ]
  %505 = icmp slt i64 %504, 32
  br i1 %505, label %506, label %550

506:                                              ; preds = %503
  br label %507

507:                                              ; preds = %510, %506
  %508 = phi i64 [ %526, %510 ], [ 0, %506 ]
  %509 = icmp slt i64 %508, 61
  br i1 %509, label %510, label %527

510:                                              ; preds = %507
  %511 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, 1
  %512 = mul i64 %504, 64
  %513 = add i64 %512, %508
  %514 = getelementptr float, ptr %511, i64 %513
  %515 = load <4 x float>, ptr %514, align 4
  %516 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %410, 1
  %517 = mul i64 %504, 64
  %518 = add i64 %517, %508
  %519 = getelementptr float, ptr %516, i64 %518
  %520 = load <4 x float>, ptr %519, align 4
  %521 = fadd <4 x float> %515, %520
  %522 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, 1
  %523 = mul i64 %504, 64
  %524 = add i64 %523, %508
  %525 = getelementptr float, ptr %522, i64 %524
  store <4 x float> %521, ptr %525, align 4
  %526 = add i64 %508, 4
  br label %507

527:                                              ; preds = %507
  br label %528

528:                                              ; preds = %531, %527
  %529 = phi i64 [ %547, %531 ], [ 64, %527 ]
  %530 = icmp slt i64 %529, 64
  br i1 %530, label %531, label %548

531:                                              ; preds = %528
  %532 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, 1
  %533 = mul nuw nsw i64 %504, 64
  %534 = add nuw nsw i64 %533, %529
  %535 = getelementptr inbounds nuw float, ptr %532, i64 %534
  %536 = load float, ptr %535, align 4
  %537 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %410, 1
  %538 = mul nuw nsw i64 %504, 64
  %539 = add nuw nsw i64 %538, %529
  %540 = getelementptr inbounds nuw float, ptr %537, i64 %539
  %541 = load float, ptr %540, align 4
  %542 = fadd float %536, %541
  %543 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, 1
  %544 = mul nuw nsw i64 %504, 64
  %545 = add nuw nsw i64 %544, %529
  %546 = getelementptr inbounds nuw float, ptr %543, i64 %545
  store float %542, ptr %546, align 4
  %547 = add i64 %529, 1
  br label %528

548:                                              ; preds = %528
  %549 = add i64 %504, 1
  br label %503

550:                                              ; preds = %503
  %551 = add i64 %117, 16
  %552 = add i64 %118, %90
  %553 = add i32 %116, 1
  br label %115

554:                                              ; preds = %115
  %555 = sext i32 %11 to i64
  %556 = mul i64 %78, %555
  %557 = add i64 %556, %80
  %558 = extractvalue { i64, ptr } %20, 1
  %559 = load ptr, ptr %558, align 8
  %560 = getelementptr ptr, ptr %558, i32 1
  %561 = load ptr, ptr %560, align 8
  %562 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %559, 0
  %563 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %562, ptr %561, 1
  %564 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %563, i64 %557, 2
  %565 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %564, i64 32, 3, 0
  %566 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %565, i64 %555, 4, 0
  %567 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %566, i64 64, 3, 1
  %568 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %567, i64 1, 4, 1
  %569 = add i64 %78, 32
  %570 = call i64 @llvm.smin.i64(i64 %569, i64 %81)
  %571 = call i64 @llvm.smax.i64(i64 %570, i64 %78)
  %572 = sub i64 %571, %78
  %573 = add i64 %80, 64
  %574 = call i64 @llvm.smin.i64(i64 %573, i64 %86)
  %575 = call i64 @llvm.smax.i64(i64 %574, i64 %80)
  %576 = sub i64 %575, %80
  %577 = call i64 @llvm.smin.i64(i64 %572, i64 32)
  %578 = call i64 @llvm.smin.i64(i64 %576, i64 64)
  %579 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, 0
  %580 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, 1
  %581 = insertvalue { ptr, ptr, i64 } poison, ptr %579, 0
  %582 = insertvalue { ptr, ptr, i64 } %581, ptr %580, 1
  %583 = insertvalue { ptr, ptr, i64 } %582, i64 0, 2
  %584 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, 2
  %585 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, 3, 0
  %586 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, 3, 1
  %587 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, 4, 0
  %588 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, 4, 1
  %589 = extractvalue { ptr, ptr, i64 } %583, 0
  %590 = extractvalue { ptr, ptr, i64 } %583, 1
  %591 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %589, 0
  %592 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %591, ptr %590, 1
  %593 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %592, i64 0, 2
  %594 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %593, i64 %577, 3, 0
  %595 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %594, i64 64, 4, 0
  %596 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %595, i64 %578, 3, 1
  %597 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %596, i64 1, 4, 1
  %598 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %568, 0
  %599 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %568, 1
  %600 = insertvalue { ptr, ptr, i64 } poison, ptr %598, 0
  %601 = insertvalue { ptr, ptr, i64 } %600, ptr %599, 1
  %602 = insertvalue { ptr, ptr, i64 } %601, i64 0, 2
  %603 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %568, 2
  %604 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %568, 3, 0
  %605 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %568, 3, 1
  %606 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %568, 4, 0
  %607 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %568, 4, 1
  %608 = extractvalue { ptr, ptr, i64 } %602, 0
  %609 = extractvalue { ptr, ptr, i64 } %602, 1
  %610 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %608, 0
  %611 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %610, ptr %609, 1
  %612 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %611, i64 %603, 2
  %613 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %612, i64 %577, 3, 0
  %614 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %613, i64 %606, 4, 0
  %615 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %614, i64 %578, 3, 1
  %616 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %615, i64 1, 4, 1
  %617 = call ptr @llvm.stacksave.p0()
  %618 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %597, ptr %618, align 8
  %619 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %618, 1
  %620 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %616, ptr %620, align 8
  %621 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %620, 1
  %622 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %619, ptr %622, align 8
  %623 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %621, ptr %623, align 8
  call void @memrefCopy(i64 4, ptr %622, ptr %623)
  call void @llvm.stackrestore.p0(ptr %617)
  ret void
}

define i32 @main() {
  %1 = call ptr @malloc(i64 2048)
  %2 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %1, 0
  %3 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %2, ptr %1, 1
  %4 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, i64 0, 2
  %5 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, i64 32, 3, 0
  %6 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, i64 16, 3, 1
  %7 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %6, i64 16, 4, 0
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, i64 1, 4, 1
  %9 = call ptr @malloc(i64 4096)
  %10 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %9, 0
  %11 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %10, ptr %9, 1
  %12 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, i64 0, 2
  %13 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, i64 16, 3, 0
  %14 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, i64 64, 3, 1
  %15 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %14, i64 64, 4, 0
  %16 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, i64 1, 4, 1
  %17 = call ptr @malloc(i64 8192)
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %17, 0
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, ptr %17, 1
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, i64 0, 2
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, i64 32, 3, 0
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, i64 64, 3, 1
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, i64 64, 4, 0
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 1, 4, 1
  %25 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, ptr %25, align 8
  %26 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %25, 1
  %27 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, ptr %27, align 8
  %28 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %27, 1
  %29 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, ptr %29, align 8
  %30 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %29, 1
  %31 = extractvalue { i64, ptr } %26, 0
  %32 = extractvalue { i64, ptr } %26, 1
  %33 = extractvalue { i64, ptr } %28, 0
  %34 = extractvalue { i64, ptr } %28, 1
  %35 = extractvalue { i64, ptr } %30, 0
  %36 = extractvalue { i64, ptr } %30, 1
  call void @matmul_kernel(i64 %31, ptr %32, i64 %33, ptr %34, i64 %35, ptr %36, i32 32, i32 64, i32 16, i32 16, i32 64, i32 64, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %37 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 0
  call void @free(ptr %37)
  %38 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 0
  call void @free(ptr %38)
  %39 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 0
  call void @free(ptr %39)
  ret i32 0
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #0

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #2

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
