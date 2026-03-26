; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

declare ptr @malloc(i64)

define void @softmax_kernel(i64 %0, ptr %1, i64 %2, ptr %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10, i32 %11, i32 %12) {
  %14 = insertvalue { i64, ptr } poison, i64 %2, 0
  %15 = insertvalue { i64, ptr } %14, ptr %3, 1
  %16 = insertvalue { i64, ptr } poison, i64 %0, 0
  %17 = insertvalue { i64, ptr } %16, ptr %1, 1
  %18 = mul i32 %10, %4
  %19 = sext i32 %18 to i64
  %20 = extractvalue { i64, ptr } %15, 1
  %21 = load ptr, ptr %20, align 8
  %22 = getelementptr ptr, ptr %20, i32 1
  %23 = load ptr, ptr %22, align 8
  %24 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %21, 0
  %25 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %24, ptr %23, 1
  %26 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, i64 %19, 2
  %27 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, i64 256, 3, 0
  %28 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %27, i64 1, 4, 0
  %29 = sext i32 %6 to i64
  %30 = call i64 @llvm.smin.i64(i64 %29, i64 256)
  %31 = call i64 @llvm.smax.i64(i64 %30, i64 0)
  %32 = call ptr @malloc(i64 1024)
  %33 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %32, 0
  %34 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, ptr %32, 1
  %35 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %34, i64 0, 2
  %36 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %35, i64 256, 3, 0
  %37 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, i64 1, 4, 0
  %38 = icmp slt i64 %31, 256
  br i1 %38, label %39, label %56

39:                                               ; preds = %13
  br label %40

40:                                               ; preds = %43, %39
  %41 = phi i64 [ %46, %43 ], [ 0, %39 ]
  %42 = icmp slt i64 %41, 253
  br i1 %42, label %43, label %47

43:                                               ; preds = %40
  %44 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %45 = getelementptr float, ptr %44, i64 %41
  store <4 x float> splat (float 0xFFF0000000000000), ptr %45, align 4
  %46 = add i64 %41, 4
  br label %40

47:                                               ; preds = %40
  br label %48

48:                                               ; preds = %51, %47
  %49 = phi i64 [ %54, %51 ], [ 256, %47 ]
  %50 = icmp slt i64 %49, 256
  br i1 %50, label %51, label %55

51:                                               ; preds = %48
  %52 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %53 = getelementptr inbounds nuw float, ptr %52, i64 %49
  store float 0xFFF0000000000000, ptr %53, align 4
  %54 = add i64 %49, 1
  br label %48

55:                                               ; preds = %48
  br label %56

56:                                               ; preds = %55, %13
  %57 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, 0
  %58 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, 1
  %59 = insertvalue { ptr, ptr, i64 } poison, ptr %57, 0
  %60 = insertvalue { ptr, ptr, i64 } %59, ptr %58, 1
  %61 = insertvalue { ptr, ptr, i64 } %60, i64 0, 2
  %62 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, 2
  %63 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, 3, 0
  %64 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, 4, 0
  %65 = extractvalue { ptr, ptr, i64 } %61, 0
  %66 = extractvalue { ptr, ptr, i64 } %61, 1
  %67 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %65, 0
  %68 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %67, ptr %66, 1
  %69 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %68, i64 %62, 2
  %70 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %69, i64 %31, 3, 0
  %71 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %70, i64 1, 4, 0
  %72 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 0
  %73 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %74 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %72, 0
  %75 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %74, ptr %73, 1
  %76 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %75, i64 0, 2
  %77 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %76, i64 %31, 3, 0
  %78 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %77, i64 1, 4, 0
  %79 = call ptr @llvm.stacksave.p0()
  %80 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %71, ptr %80, align 8
  %81 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %80, 1
  %82 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %78, ptr %82, align 8
  %83 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %82, 1
  %84 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %81, ptr %84, align 8
  %85 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %83, ptr %85, align 8
  call void @memrefCopy(i64 4, ptr %84, ptr %85)
  call void @llvm.stackrestore.p0(ptr %79)
  %86 = call ptr @malloc(i64 68)
  %87 = ptrtoint ptr %86 to i64
  %88 = add i64 %87, 63
  %89 = urem i64 %88, 64
  %90 = sub i64 %88, %89
  %91 = inttoptr i64 %90 to ptr
  %92 = insertvalue { ptr, ptr, i64 } poison, ptr %86, 0
  %93 = insertvalue { ptr, ptr, i64 } %92, ptr %91, 1
  %94 = insertvalue { ptr, ptr, i64 } %93, i64 0, 2
  %95 = extractvalue { ptr, ptr, i64 } %94, 1
  store float 0xFFF0000000000000, ptr %95, align 4
  %96 = alloca float, i64 1, align 4
  %97 = insertvalue { ptr, ptr, i64 } poison, ptr %96, 0
  %98 = insertvalue { ptr, ptr, i64 } %97, ptr %96, 1
  %99 = insertvalue { ptr, ptr, i64 } %98, i64 0, 2
  %100 = extractvalue { ptr, ptr, i64 } %94, 1
  %101 = load float, ptr %100, align 4
  %102 = extractvalue { ptr, ptr, i64 } %99, 1
  store float %101, ptr %102, align 4
  br label %103

103:                                              ; preds = %106, %56
  %104 = phi i64 [ %115, %106 ], [ 0, %56 ]
  %105 = icmp slt i64 %104, 253
  br i1 %105, label %106, label %116

106:                                              ; preds = %103
  %107 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %108 = getelementptr float, ptr %107, i64 %104
  %109 = load <4 x float>, ptr %108, align 4
  %110 = extractvalue { ptr, ptr, i64 } %99, 1
  %111 = load float, ptr %110, align 4
  %112 = call float @llvm.vector.reduce.fmax.v4f32(<4 x float> %109)
  %113 = call float @llvm.maxnum.f32(float %112, float %111)
  %114 = extractvalue { ptr, ptr, i64 } %99, 1
  store float %113, ptr %114, align 4
  %115 = add i64 %104, 4
  br label %103

116:                                              ; preds = %103
  br label %117

117:                                              ; preds = %120, %116
  %118 = phi i64 [ %128, %120 ], [ 256, %116 ]
  %119 = icmp slt i64 %118, 256
  br i1 %119, label %120, label %129

120:                                              ; preds = %117
  %121 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %122 = getelementptr inbounds nuw float, ptr %121, i64 %118
  %123 = load float, ptr %122, align 4
  %124 = extractvalue { ptr, ptr, i64 } %99, 1
  %125 = load float, ptr %124, align 4
  %126 = call float @llvm.maxnum.f32(float %123, float %125)
  %127 = extractvalue { ptr, ptr, i64 } %99, 1
  store float %126, ptr %127, align 4
  %128 = add i64 %118, 1
  br label %117

129:                                              ; preds = %117
  %130 = extractvalue { ptr, ptr, i64 } %99, 1
  %131 = load float, ptr %130, align 4
  %132 = extractvalue { ptr, ptr, i64 } %94, 1
  store float %131, ptr %132, align 4
  %133 = extractvalue { ptr, ptr, i64 } %94, 1
  %134 = load float, ptr %133, align 4
  %135 = call ptr @malloc(i64 1088)
  %136 = ptrtoint ptr %135 to i64
  %137 = add i64 %136, 63
  %138 = urem i64 %137, 64
  %139 = sub i64 %137, %138
  %140 = inttoptr i64 %139 to ptr
  %141 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %135, 0
  %142 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %141, ptr %140, 1
  %143 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %142, i64 0, 2
  %144 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %143, i64 256, 3, 0
  %145 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %144, i64 1, 4, 0
  br label %146

146:                                              ; preds = %149, %129
  %147 = phi i64 [ %154, %149 ], [ 0, %129 ]
  %148 = icmp slt i64 %147, 253
  br i1 %148, label %149, label %155

149:                                              ; preds = %146
  %150 = insertelement <4 x float> poison, float %134, i32 0
  %151 = shufflevector <4 x float> %150, <4 x float> poison, <4 x i32> zeroinitializer
  %152 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %145, 1
  %153 = getelementptr float, ptr %152, i64 %147
  store <4 x float> %151, ptr %153, align 4
  %154 = add i64 %147, 4
  br label %146

155:                                              ; preds = %146
  br label %156

156:                                              ; preds = %159, %155
  %157 = phi i64 [ %162, %159 ], [ 256, %155 ]
  %158 = icmp slt i64 %157, 256
  br i1 %158, label %159, label %163

159:                                              ; preds = %156
  %160 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %145, 1
  %161 = getelementptr inbounds nuw float, ptr %160, i64 %157
  store float %134, ptr %161, align 4
  %162 = add i64 %157, 1
  br label %156

163:                                              ; preds = %156
  br label %164

164:                                              ; preds = %167, %163
  %165 = phi i64 [ %177, %167 ], [ 0, %163 ]
  %166 = icmp slt i64 %165, 253
  br i1 %166, label %167, label %178

167:                                              ; preds = %164
  %168 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %169 = getelementptr float, ptr %168, i64 %165
  %170 = load <4 x float>, ptr %169, align 4
  %171 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %145, 1
  %172 = getelementptr float, ptr %171, i64 %165
  %173 = load <4 x float>, ptr %172, align 4
  %174 = fsub <4 x float> %170, %173
  %175 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %176 = getelementptr float, ptr %175, i64 %165
  store <4 x float> %174, ptr %176, align 4
  %177 = add i64 %165, 4
  br label %164

178:                                              ; preds = %164
  br label %179

179:                                              ; preds = %182, %178
  %180 = phi i64 [ %192, %182 ], [ 256, %178 ]
  %181 = icmp slt i64 %180, 256
  br i1 %181, label %182, label %193

182:                                              ; preds = %179
  %183 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %184 = getelementptr inbounds nuw float, ptr %183, i64 %180
  %185 = load float, ptr %184, align 4
  %186 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %145, 1
  %187 = getelementptr inbounds nuw float, ptr %186, i64 %180
  %188 = load float, ptr %187, align 4
  %189 = fsub float %185, %188
  %190 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %191 = getelementptr inbounds nuw float, ptr %190, i64 %180
  store float %189, ptr %191, align 4
  %192 = add i64 %180, 1
  br label %179

193:                                              ; preds = %179
  br label %194

194:                                              ; preds = %197, %193
  %195 = phi i64 [ %204, %197 ], [ 0, %193 ]
  %196 = icmp slt i64 %195, 253
  br i1 %196, label %197, label %205

197:                                              ; preds = %194
  %198 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %199 = getelementptr float, ptr %198, i64 %195
  %200 = load <4 x float>, ptr %199, align 4
  %201 = call <4 x float> @llvm.exp.v4f32(<4 x float> %200)
  %202 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %203 = getelementptr float, ptr %202, i64 %195
  store <4 x float> %201, ptr %203, align 4
  %204 = add i64 %195, 4
  br label %194

205:                                              ; preds = %194
  br label %206

206:                                              ; preds = %209, %205
  %207 = phi i64 [ %216, %209 ], [ 256, %205 ]
  %208 = icmp slt i64 %207, 256
  br i1 %208, label %209, label %217

209:                                              ; preds = %206
  %210 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %211 = getelementptr inbounds nuw float, ptr %210, i64 %207
  %212 = load float, ptr %211, align 4
  %213 = call float @llvm.exp.f32(float %212)
  %214 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %215 = getelementptr inbounds nuw float, ptr %214, i64 %207
  store float %213, ptr %215, align 4
  %216 = add i64 %207, 1
  br label %206

217:                                              ; preds = %206
  %218 = call ptr @malloc(i64 68)
  %219 = ptrtoint ptr %218 to i64
  %220 = add i64 %219, 63
  %221 = urem i64 %220, 64
  %222 = sub i64 %220, %221
  %223 = inttoptr i64 %222 to ptr
  %224 = insertvalue { ptr, ptr, i64 } poison, ptr %218, 0
  %225 = insertvalue { ptr, ptr, i64 } %224, ptr %223, 1
  %226 = insertvalue { ptr, ptr, i64 } %225, i64 0, 2
  %227 = extractvalue { ptr, ptr, i64 } %226, 1
  store float 0.000000e+00, ptr %227, align 4
  %228 = alloca float, i64 1, align 4
  %229 = insertvalue { ptr, ptr, i64 } poison, ptr %228, 0
  %230 = insertvalue { ptr, ptr, i64 } %229, ptr %228, 1
  %231 = insertvalue { ptr, ptr, i64 } %230, i64 0, 2
  %232 = extractvalue { ptr, ptr, i64 } %226, 1
  %233 = load float, ptr %232, align 4
  %234 = extractvalue { ptr, ptr, i64 } %231, 1
  store float %233, ptr %234, align 4
  br label %235

235:                                              ; preds = %238, %217
  %236 = phi i64 [ %246, %238 ], [ 0, %217 ]
  %237 = icmp slt i64 %236, 253
  br i1 %237, label %238, label %247

238:                                              ; preds = %235
  %239 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %240 = getelementptr float, ptr %239, i64 %236
  %241 = load <4 x float>, ptr %240, align 4
  %242 = extractvalue { ptr, ptr, i64 } %231, 1
  %243 = load float, ptr %242, align 4
  %244 = call float @llvm.vector.reduce.fadd.v4f32(float %243, <4 x float> %241)
  %245 = extractvalue { ptr, ptr, i64 } %231, 1
  store float %244, ptr %245, align 4
  %246 = add i64 %236, 4
  br label %235

247:                                              ; preds = %235
  br label %248

248:                                              ; preds = %251, %247
  %249 = phi i64 [ %259, %251 ], [ 256, %247 ]
  %250 = icmp slt i64 %249, 256
  br i1 %250, label %251, label %260

251:                                              ; preds = %248
  %252 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %253 = getelementptr inbounds nuw float, ptr %252, i64 %249
  %254 = load float, ptr %253, align 4
  %255 = extractvalue { ptr, ptr, i64 } %231, 1
  %256 = load float, ptr %255, align 4
  %257 = fadd float %254, %256
  %258 = extractvalue { ptr, ptr, i64 } %231, 1
  store float %257, ptr %258, align 4
  %259 = add i64 %249, 1
  br label %248

260:                                              ; preds = %248
  %261 = extractvalue { ptr, ptr, i64 } %231, 1
  %262 = load float, ptr %261, align 4
  %263 = extractvalue { ptr, ptr, i64 } %226, 1
  store float %262, ptr %263, align 4
  %264 = extractvalue { ptr, ptr, i64 } %226, 1
  %265 = load float, ptr %264, align 4
  br label %266

266:                                              ; preds = %269, %260
  %267 = phi i64 [ %274, %269 ], [ 0, %260 ]
  %268 = icmp slt i64 %267, 253
  br i1 %268, label %269, label %275

269:                                              ; preds = %266
  %270 = insertelement <4 x float> poison, float %265, i32 0
  %271 = shufflevector <4 x float> %270, <4 x float> poison, <4 x i32> zeroinitializer
  %272 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %145, 1
  %273 = getelementptr float, ptr %272, i64 %267
  store <4 x float> %271, ptr %273, align 4
  %274 = add i64 %267, 4
  br label %266

275:                                              ; preds = %266
  br label %276

276:                                              ; preds = %279, %275
  %277 = phi i64 [ %282, %279 ], [ 256, %275 ]
  %278 = icmp slt i64 %277, 256
  br i1 %278, label %279, label %283

279:                                              ; preds = %276
  %280 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %145, 1
  %281 = getelementptr inbounds nuw float, ptr %280, i64 %277
  store float %265, ptr %281, align 4
  %282 = add i64 %277, 1
  br label %276

283:                                              ; preds = %276
  br label %284

284:                                              ; preds = %287, %283
  %285 = phi i64 [ %297, %287 ], [ 0, %283 ]
  %286 = icmp slt i64 %285, 253
  br i1 %286, label %287, label %298

287:                                              ; preds = %284
  %288 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %289 = getelementptr float, ptr %288, i64 %285
  %290 = load <4 x float>, ptr %289, align 4
  %291 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %145, 1
  %292 = getelementptr float, ptr %291, i64 %285
  %293 = load <4 x float>, ptr %292, align 4
  %294 = fdiv <4 x float> %290, %293
  %295 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %296 = getelementptr float, ptr %295, i64 %285
  store <4 x float> %294, ptr %296, align 4
  %297 = add i64 %285, 4
  br label %284

298:                                              ; preds = %284
  br label %299

299:                                              ; preds = %302, %298
  %300 = phi i64 [ %312, %302 ], [ 256, %298 ]
  %301 = icmp slt i64 %300, 256
  br i1 %301, label %302, label %313

302:                                              ; preds = %299
  %303 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %304 = getelementptr inbounds nuw float, ptr %303, i64 %300
  %305 = load float, ptr %304, align 4
  %306 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %145, 1
  %307 = getelementptr inbounds nuw float, ptr %306, i64 %300
  %308 = load float, ptr %307, align 4
  %309 = fdiv float %305, %308
  %310 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %311 = getelementptr inbounds nuw float, ptr %310, i64 %300
  store float %309, ptr %311, align 4
  %312 = add i64 %300, 1
  br label %299

313:                                              ; preds = %299
  %314 = mul i32 %10, %5
  %315 = sext i32 %314 to i64
  %316 = extractvalue { i64, ptr } %17, 1
  %317 = load ptr, ptr %316, align 8
  %318 = getelementptr ptr, ptr %316, i32 1
  %319 = load ptr, ptr %318, align 8
  %320 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %317, 0
  %321 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %320, ptr %319, 1
  %322 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %321, i64 %315, 2
  %323 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %322, i64 256, 3, 0
  %324 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %323, i64 1, 4, 0
  %325 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %324, 0
  %326 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %324, 1
  %327 = insertvalue { ptr, ptr, i64 } poison, ptr %325, 0
  %328 = insertvalue { ptr, ptr, i64 } %327, ptr %326, 1
  %329 = insertvalue { ptr, ptr, i64 } %328, i64 0, 2
  %330 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %324, 2
  %331 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %324, 3, 0
  %332 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %324, 4, 0
  %333 = extractvalue { ptr, ptr, i64 } %329, 0
  %334 = extractvalue { ptr, ptr, i64 } %329, 1
  %335 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %333, 0
  %336 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %335, ptr %334, 1
  %337 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %336, i64 %330, 2
  %338 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %337, i64 %31, 3, 0
  %339 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %338, i64 1, 4, 0
  %340 = call ptr @llvm.stacksave.p0()
  %341 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %78, ptr %341, align 8
  %342 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %341, 1
  %343 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %339, ptr %343, align 8
  %344 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %343, 1
  %345 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %342, ptr %345, align 8
  %346 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %344, ptr %346, align 8
  call void @memrefCopy(i64 4, ptr %345, ptr %346)
  call void @llvm.stackrestore.p0(ptr %340)
  ret void
}

define i32 @main() {
  %1 = call ptr @malloc(i64 1024)
  %2 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %1, 0
  %3 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2, ptr %1, 1
  %4 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %3, i64 0, 2
  %5 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, i64 256, 3, 0
  %6 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, i64 1, 4, 0
  %7 = call ptr @malloc(i64 1024)
  %8 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %7, 0
  %9 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, ptr %7, 1
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, i64 0, 2
  %11 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, i64 256, 3, 0
  %12 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, i64 1, 4, 0
  %13 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, ptr %13, align 8
  %14 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %13, 1
  %15 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, ptr %15, align 8
  %16 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %15, 1
  %17 = extractvalue { i64, ptr } %14, 0
  %18 = extractvalue { i64, ptr } %14, 1
  %19 = extractvalue { i64, ptr } %16, 0
  %20 = extractvalue { i64, ptr } %16, 1
  call void @softmax_kernel(i64 %17, ptr %18, i64 %19, ptr %20, i32 256, i32 256, i32 256, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %21 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 0
  call void @free(ptr %21)
  %22 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 0
  call void @free(ptr %22)
  ret i32 0
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.vector.reduce.fadd.v4f32(float, <4 x float>) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp.f32(float) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.exp.v4f32(<4 x float>) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.vector.reduce.fmax.v4f32(<4 x float>) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
