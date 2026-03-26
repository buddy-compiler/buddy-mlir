; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

declare ptr @malloc(i64)

define void @_layer_norm_fwd_fused(i64 %0, ptr %1, i64 %2, ptr %3, i64 %4, ptr %5, i64 %6, ptr %7, i64 %8, ptr %9, i64 %10, ptr %11, i32 %12, i32 %13, float %14, i32 %15, i32 %16, i32 %17, i32 %18, i32 %19, i32 %20) {
  %22 = insertvalue { i64, ptr } poison, i64 %10, 0
  %23 = insertvalue { i64, ptr } %22, ptr %11, 1
  %24 = insertvalue { i64, ptr } poison, i64 %8, 0
  %25 = insertvalue { i64, ptr } %24, ptr %9, 1
  %26 = insertvalue { i64, ptr } poison, i64 %6, 0
  %27 = insertvalue { i64, ptr } %26, ptr %7, 1
  %28 = insertvalue { i64, ptr } poison, i64 %4, 0
  %29 = insertvalue { i64, ptr } %28, ptr %5, 1
  %30 = insertvalue { i64, ptr } poison, i64 %2, 0
  %31 = insertvalue { i64, ptr } %30, ptr %3, 1
  %32 = insertvalue { i64, ptr } poison, i64 %0, 0
  %33 = insertvalue { i64, ptr } %32, ptr %1, 1
  %34 = call ptr @malloc(i64 2112)
  %35 = ptrtoint ptr %34 to i64
  %36 = add i64 %35, 63
  %37 = urem i64 %36, 64
  %38 = sub i64 %36, %37
  %39 = inttoptr i64 %38 to ptr
  %40 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %34, 0
  %41 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, ptr %39, 1
  %42 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %41, i64 0, 2
  %43 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %42, i64 512, 3, 0
  %44 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %43, i64 1, 4, 0
  %45 = call ptr @malloc(i64 2112)
  %46 = ptrtoint ptr %45 to i64
  %47 = add i64 %46, 63
  %48 = urem i64 %47, 64
  %49 = sub i64 %47, %48
  %50 = inttoptr i64 %49 to ptr
  %51 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %45, 0
  %52 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %51, ptr %50, 1
  %53 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %52, i64 0, 2
  %54 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %53, i64 512, 3, 0
  %55 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %54, i64 1, 4, 0
  br label %56

56:                                               ; preds = %59, %21
  %57 = phi i64 [ %62, %59 ], [ 0, %21 ]
  %58 = icmp slt i64 %57, 509
  br i1 %58, label %59, label %63

59:                                               ; preds = %56
  %60 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 1
  %61 = getelementptr float, ptr %60, i64 %57
  store <4 x float> zeroinitializer, ptr %61, align 4
  %62 = add i64 %57, 4
  br label %56

63:                                               ; preds = %56
  br label %64

64:                                               ; preds = %67, %63
  %65 = phi i64 [ %70, %67 ], [ 512, %63 ]
  %66 = icmp slt i64 %65, 512
  br i1 %66, label %67, label %71

67:                                               ; preds = %64
  %68 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 1
  %69 = getelementptr inbounds nuw float, ptr %68, i64 %65
  store float 0.000000e+00, ptr %69, align 4
  %70 = add i64 %65, 1
  br label %64

71:                                               ; preds = %64
  %72 = mul i32 %18, %12
  %73 = sext i32 %72 to i64
  %74 = call ptr @malloc(i64 2112)
  %75 = ptrtoint ptr %74 to i64
  %76 = add i64 %75, 63
  %77 = urem i64 %76, 64
  %78 = sub i64 %76, %77
  %79 = inttoptr i64 %78 to ptr
  %80 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %74, 0
  %81 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %80, ptr %79, 1
  %82 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, i64 0, 2
  %83 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, i64 512, 3, 0
  %84 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %83, i64 1, 4, 0
  %85 = call ptr @malloc(i64 2112)
  %86 = ptrtoint ptr %85 to i64
  %87 = add i64 %86, 63
  %88 = urem i64 %87, 64
  %89 = sub i64 %87, %88
  %90 = inttoptr i64 %89 to ptr
  %91 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %85, 0
  %92 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %91, ptr %90, 1
  %93 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %92, i64 0, 2
  %94 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %93, i64 512, 3, 0
  %95 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %94, i64 1, 4, 0
  br label %96

96:                                               ; preds = %99, %71
  %97 = phi i64 [ %103, %99 ], [ 0, %71 ]
  %98 = icmp slt i64 %97, 512
  br i1 %98, label %99, label %104

99:                                               ; preds = %96
  %100 = trunc i64 %97 to i32
  %101 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %95, 1
  %102 = getelementptr inbounds nuw i32, ptr %101, i64 %97
  store i32 %100, ptr %102, align 4
  %103 = add i64 %97, 1
  br label %96

104:                                              ; preds = %96
  %105 = call ptr @malloc(i64 2112)
  %106 = ptrtoint ptr %105 to i64
  %107 = add i64 %106, 63
  %108 = urem i64 %107, 64
  %109 = sub i64 %107, %108
  %110 = inttoptr i64 %109 to ptr
  %111 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %105, 0
  %112 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %111, ptr %110, 1
  %113 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %112, i64 0, 2
  %114 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %113, i64 512, 3, 0
  %115 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %114, i64 1, 4, 0
  br label %116

116:                                              ; preds = %119, %104
  %117 = phi i64 [ %124, %119 ], [ 0, %104 ]
  %118 = icmp slt i64 %117, 509
  br i1 %118, label %119, label %125

119:                                              ; preds = %116
  %120 = insertelement <4 x i32> poison, i32 %13, i32 0
  %121 = shufflevector <4 x i32> %120, <4 x i32> poison, <4 x i32> zeroinitializer
  %122 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %115, 1
  %123 = getelementptr i32, ptr %122, i64 %117
  store <4 x i32> %121, ptr %123, align 4
  %124 = add i64 %117, 4
  br label %116

125:                                              ; preds = %116
  br label %126

126:                                              ; preds = %129, %125
  %127 = phi i64 [ %132, %129 ], [ 512, %125 ]
  %128 = icmp slt i64 %127, 512
  br i1 %128, label %129, label %133

129:                                              ; preds = %126
  %130 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %115, 1
  %131 = getelementptr inbounds nuw i32, ptr %130, i64 %127
  store i32 %13, ptr %131, align 4
  %132 = add i64 %127, 1
  br label %126

133:                                              ; preds = %126
  %134 = call ptr @malloc(i64 2112)
  %135 = ptrtoint ptr %134 to i64
  %136 = add i64 %135, 63
  %137 = urem i64 %136, 64
  %138 = sub i64 %136, %137
  %139 = inttoptr i64 %138 to ptr
  %140 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %134, 0
  %141 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %140, ptr %139, 1
  %142 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %141, i64 0, 2
  %143 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %142, i64 512, 3, 0
  %144 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %143, i64 1, 4, 0
  %145 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 3, 0
  %146 = mul i64 1, %145
  %147 = mul i64 %146, 4
  %148 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 1
  %149 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 2
  %150 = getelementptr float, ptr %148, i64 %149
  %151 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %144, 1
  %152 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %144, 2
  %153 = getelementptr float, ptr %151, i64 %152
  call void @llvm.memcpy.p0.p0.i64(ptr %153, ptr %150, i64 %147, i1 false)
  br label %154

154:                                              ; preds = %282, %133
  %155 = phi i32 [ %283, %282 ], [ 0, %133 ]
  %156 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %156, %282 ], [ %144, %133 ]
  %157 = icmp slt i32 %155, %13
  br i1 %157, label %158, label %284

158:                                              ; preds = %154
  %159 = sext i32 %155 to i64
  %160 = add i64 %73, %159
  %161 = extractvalue { i64, ptr } %33, 1
  %162 = load ptr, ptr %161, align 8
  %163 = getelementptr ptr, ptr %161, i32 1
  %164 = load ptr, ptr %163, align 8
  %165 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %162, 0
  %166 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %165, ptr %164, 1
  %167 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %166, i64 %160, 2
  %168 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %167, i64 512, 3, 0
  %169 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %168, i64 1, 4, 0
  %170 = add i64 %159, 512
  %171 = sext i32 %13 to i64
  %172 = call i64 @llvm.smin.i64(i64 %170, i64 %171)
  %173 = call i64 @llvm.smax.i64(i64 %172, i64 %159)
  %174 = sub i64 %173, %159
  %175 = call ptr @malloc(i64 1024)
  %176 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %175, 0
  %177 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %176, ptr %175, 1
  %178 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %177, i64 0, 2
  %179 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %178, i64 512, 3, 0
  %180 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %179, i64 1, 4, 0
  %181 = icmp slt i64 %174, 512
  br i1 %181, label %182, label %199

182:                                              ; preds = %158
  br label %183

183:                                              ; preds = %186, %182
  %184 = phi i64 [ %189, %186 ], [ 0, %182 ]
  %185 = icmp slt i64 %184, 509
  br i1 %185, label %186, label %190

186:                                              ; preds = %183
  %187 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %180, 1
  %188 = getelementptr half, ptr %187, i64 %184
  store <4 x half> zeroinitializer, ptr %188, align 2
  %189 = add i64 %184, 4
  br label %183

190:                                              ; preds = %183
  br label %191

191:                                              ; preds = %194, %190
  %192 = phi i64 [ %197, %194 ], [ 512, %190 ]
  %193 = icmp slt i64 %192, 512
  br i1 %193, label %194, label %198

194:                                              ; preds = %191
  %195 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %180, 1
  %196 = getelementptr inbounds nuw half, ptr %195, i64 %192
  store half 0xH0000, ptr %196, align 2
  %197 = add i64 %192, 1
  br label %191

198:                                              ; preds = %191
  br label %199

199:                                              ; preds = %198, %158
  %200 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %169, 0
  %201 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %169, 1
  %202 = insertvalue { ptr, ptr, i64 } poison, ptr %200, 0
  %203 = insertvalue { ptr, ptr, i64 } %202, ptr %201, 1
  %204 = insertvalue { ptr, ptr, i64 } %203, i64 0, 2
  %205 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %169, 2
  %206 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %169, 3, 0
  %207 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %169, 4, 0
  %208 = extractvalue { ptr, ptr, i64 } %204, 0
  %209 = extractvalue { ptr, ptr, i64 } %204, 1
  %210 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %208, 0
  %211 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %210, ptr %209, 1
  %212 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %211, i64 %205, 2
  %213 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %212, i64 %174, 3, 0
  %214 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %213, i64 1, 4, 0
  %215 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %180, 0
  %216 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %180, 1
  %217 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %215, 0
  %218 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %217, ptr %216, 1
  %219 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %218, i64 0, 2
  %220 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %219, i64 %174, 3, 0
  %221 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %220, i64 1, 4, 0
  %222 = call ptr @llvm.stacksave.p0()
  %223 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %214, ptr %223, align 8
  %224 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %223, 1
  %225 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %221, ptr %225, align 8
  %226 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %225, 1
  %227 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %224, ptr %227, align 8
  %228 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %226, ptr %228, align 8
  call void @memrefCopy(i64 2, ptr %227, ptr %228)
  call void @llvm.stackrestore.p0(ptr %222)
  br label %229

229:                                              ; preds = %232, %199
  %230 = phi i64 [ %239, %232 ], [ 0, %199 ]
  %231 = icmp slt i64 %230, 509
  br i1 %231, label %232, label %240

232:                                              ; preds = %229
  %233 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %180, 1
  %234 = getelementptr half, ptr %233, i64 %230
  %235 = load <4 x half>, ptr %234, align 2
  %236 = fpext <4 x half> %235 to <4 x float>
  %237 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %238 = getelementptr float, ptr %237, i64 %230
  store <4 x float> %236, ptr %238, align 4
  %239 = add i64 %230, 4
  br label %229

240:                                              ; preds = %229
  br label %241

241:                                              ; preds = %244, %240
  %242 = phi i64 [ %251, %244 ], [ 512, %240 ]
  %243 = icmp slt i64 %242, 512
  br i1 %243, label %244, label %252

244:                                              ; preds = %241
  %245 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %180, 1
  %246 = getelementptr inbounds nuw half, ptr %245, i64 %242
  %247 = load half, ptr %246, align 2
  %248 = fpext half %247 to float
  %249 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %250 = getelementptr inbounds nuw float, ptr %249, i64 %242
  store float %248, ptr %250, align 4
  %251 = add i64 %242, 1
  br label %241

252:                                              ; preds = %241
  br label %253

253:                                              ; preds = %256, %252
  %254 = phi i64 [ %266, %256 ], [ 0, %252 ]
  %255 = icmp slt i64 %254, 509
  br i1 %255, label %256, label %267

256:                                              ; preds = %253
  %257 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %156, 1
  %258 = getelementptr float, ptr %257, i64 %254
  %259 = load <4 x float>, ptr %258, align 4
  %260 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %261 = getelementptr float, ptr %260, i64 %254
  %262 = load <4 x float>, ptr %261, align 4
  %263 = fadd <4 x float> %259, %262
  %264 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %156, 1
  %265 = getelementptr float, ptr %264, i64 %254
  store <4 x float> %263, ptr %265, align 4
  %266 = add i64 %254, 4
  br label %253

267:                                              ; preds = %253
  br label %268

268:                                              ; preds = %271, %267
  %269 = phi i64 [ %281, %271 ], [ 512, %267 ]
  %270 = icmp slt i64 %269, 512
  br i1 %270, label %271, label %282

271:                                              ; preds = %268
  %272 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %156, 1
  %273 = getelementptr inbounds nuw float, ptr %272, i64 %269
  %274 = load float, ptr %273, align 4
  %275 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %276 = getelementptr inbounds nuw float, ptr %275, i64 %269
  %277 = load float, ptr %276, align 4
  %278 = fadd float %274, %277
  %279 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %156, 1
  %280 = getelementptr inbounds nuw float, ptr %279, i64 %269
  store float %278, ptr %280, align 4
  %281 = add i64 %269, 1
  br label %268

282:                                              ; preds = %268
  %283 = add i32 %155, 512
  br label %154

284:                                              ; preds = %154
  %285 = call ptr @malloc(i64 68)
  %286 = ptrtoint ptr %285 to i64
  %287 = add i64 %286, 63
  %288 = urem i64 %287, 64
  %289 = sub i64 %287, %288
  %290 = inttoptr i64 %289 to ptr
  %291 = insertvalue { ptr, ptr, i64 } poison, ptr %285, 0
  %292 = insertvalue { ptr, ptr, i64 } %291, ptr %290, 1
  %293 = insertvalue { ptr, ptr, i64 } %292, i64 0, 2
  %294 = extractvalue { ptr, ptr, i64 } %293, 1
  store float 0.000000e+00, ptr %294, align 4
  %295 = alloca float, i64 1, align 4
  %296 = insertvalue { ptr, ptr, i64 } poison, ptr %295, 0
  %297 = insertvalue { ptr, ptr, i64 } %296, ptr %295, 1
  %298 = insertvalue { ptr, ptr, i64 } %297, i64 0, 2
  %299 = extractvalue { ptr, ptr, i64 } %293, 1
  %300 = load float, ptr %299, align 4
  %301 = extractvalue { ptr, ptr, i64 } %298, 1
  store float %300, ptr %301, align 4
  br label %302

302:                                              ; preds = %305, %284
  %303 = phi i64 [ %313, %305 ], [ 0, %284 ]
  %304 = icmp slt i64 %303, 509
  br i1 %304, label %305, label %314

305:                                              ; preds = %302
  %306 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %156, 1
  %307 = getelementptr float, ptr %306, i64 %303
  %308 = load <4 x float>, ptr %307, align 4
  %309 = extractvalue { ptr, ptr, i64 } %298, 1
  %310 = load float, ptr %309, align 4
  %311 = call float @llvm.vector.reduce.fadd.v4f32(float %310, <4 x float> %308)
  %312 = extractvalue { ptr, ptr, i64 } %298, 1
  store float %311, ptr %312, align 4
  %313 = add i64 %303, 4
  br label %302

314:                                              ; preds = %302
  br label %315

315:                                              ; preds = %318, %314
  %316 = phi i64 [ %326, %318 ], [ 512, %314 ]
  %317 = icmp slt i64 %316, 512
  br i1 %317, label %318, label %327

318:                                              ; preds = %315
  %319 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %156, 1
  %320 = getelementptr inbounds nuw float, ptr %319, i64 %316
  %321 = load float, ptr %320, align 4
  %322 = extractvalue { ptr, ptr, i64 } %298, 1
  %323 = load float, ptr %322, align 4
  %324 = fadd float %321, %323
  %325 = extractvalue { ptr, ptr, i64 } %298, 1
  store float %324, ptr %325, align 4
  %326 = add i64 %316, 1
  br label %315

327:                                              ; preds = %315
  %328 = extractvalue { ptr, ptr, i64 } %298, 1
  %329 = load float, ptr %328, align 4
  %330 = extractvalue { ptr, ptr, i64 } %293, 1
  store float %329, ptr %330, align 4
  %331 = extractvalue { ptr, ptr, i64 } %293, 1
  %332 = load float, ptr %331, align 4
  %333 = sitofp i32 %13 to float
  %334 = fdiv float %332, %333
  %335 = call ptr @malloc(i64 2112)
  %336 = ptrtoint ptr %335 to i64
  %337 = add i64 %336, 63
  %338 = urem i64 %337, 64
  %339 = sub i64 %337, %338
  %340 = inttoptr i64 %339 to ptr
  %341 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %335, 0
  %342 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %341, ptr %340, 1
  %343 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %342, i64 0, 2
  %344 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %343, i64 512, 3, 0
  %345 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %344, i64 1, 4, 0
  br label %346

346:                                              ; preds = %349, %327
  %347 = phi i64 [ %354, %349 ], [ 0, %327 ]
  %348 = icmp slt i64 %347, 509
  br i1 %348, label %349, label %355

349:                                              ; preds = %346
  %350 = insertelement <4 x float> poison, float %334, i32 0
  %351 = shufflevector <4 x float> %350, <4 x float> poison, <4 x i32> zeroinitializer
  %352 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %345, 1
  %353 = getelementptr float, ptr %352, i64 %347
  store <4 x float> %351, ptr %353, align 4
  %354 = add i64 %347, 4
  br label %346

355:                                              ; preds = %346
  br label %356

356:                                              ; preds = %359, %355
  %357 = phi i64 [ %362, %359 ], [ 512, %355 ]
  %358 = icmp slt i64 %357, 512
  br i1 %358, label %359, label %363

359:                                              ; preds = %356
  %360 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %345, 1
  %361 = getelementptr inbounds nuw float, ptr %360, i64 %357
  store float %334, ptr %361, align 4
  %362 = add i64 %357, 1
  br label %356

363:                                              ; preds = %356
  %364 = call ptr @malloc(i64 2112)
  %365 = ptrtoint ptr %364 to i64
  %366 = add i64 %365, 63
  %367 = urem i64 %366, 64
  %368 = sub i64 %366, %367
  %369 = inttoptr i64 %368 to ptr
  %370 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %364, 0
  %371 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %370, ptr %369, 1
  %372 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %371, i64 0, 2
  %373 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %372, i64 512, 3, 0
  %374 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %373, i64 1, 4, 0
  %375 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 3, 0
  %376 = mul i64 1, %375
  %377 = mul i64 %376, 4
  %378 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 1
  %379 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 2
  %380 = getelementptr float, ptr %378, i64 %379
  %381 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %374, 1
  %382 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %374, 2
  %383 = getelementptr float, ptr %381, i64 %382
  call void @llvm.memcpy.p0.p0.i64(ptr %383, ptr %380, i64 %377, i1 false)
  br label %384

384:                                              ; preds = %691, %363
  %385 = phi i32 [ %692, %691 ], [ 0, %363 ]
  %386 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %386, %691 ], [ %374, %363 ]
  %387 = icmp slt i32 %385, %13
  br i1 %387, label %388, label %693

388:                                              ; preds = %384
  br label %389

389:                                              ; preds = %392, %388
  %390 = phi i64 [ %397, %392 ], [ 0, %388 ]
  %391 = icmp slt i64 %390, 509
  br i1 %391, label %392, label %398

392:                                              ; preds = %389
  %393 = insertelement <4 x i32> poison, i32 %385, i32 0
  %394 = shufflevector <4 x i32> %393, <4 x i32> poison, <4 x i32> zeroinitializer
  %395 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %84, 1
  %396 = getelementptr i32, ptr %395, i64 %390
  store <4 x i32> %394, ptr %396, align 4
  %397 = add i64 %390, 4
  br label %389

398:                                              ; preds = %389
  br label %399

399:                                              ; preds = %402, %398
  %400 = phi i64 [ %405, %402 ], [ 512, %398 ]
  %401 = icmp slt i64 %400, 512
  br i1 %401, label %402, label %406

402:                                              ; preds = %399
  %403 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %84, 1
  %404 = getelementptr inbounds nuw i32, ptr %403, i64 %400
  store i32 %385, ptr %404, align 4
  %405 = add i64 %400, 1
  br label %399

406:                                              ; preds = %399
  br label %407

407:                                              ; preds = %410, %406
  %408 = phi i64 [ %420, %410 ], [ 0, %406 ]
  %409 = icmp slt i64 %408, 509
  br i1 %409, label %410, label %421

410:                                              ; preds = %407
  %411 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %84, 1
  %412 = getelementptr i32, ptr %411, i64 %408
  %413 = load <4 x i32>, ptr %412, align 4
  %414 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %95, 1
  %415 = getelementptr i32, ptr %414, i64 %408
  %416 = load <4 x i32>, ptr %415, align 4
  %417 = add <4 x i32> %413, %416
  %418 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %84, 1
  %419 = getelementptr i32, ptr %418, i64 %408
  store <4 x i32> %417, ptr %419, align 4
  %420 = add i64 %408, 4
  br label %407

421:                                              ; preds = %407
  br label %422

422:                                              ; preds = %425, %421
  %423 = phi i64 [ %435, %425 ], [ 512, %421 ]
  %424 = icmp slt i64 %423, 512
  br i1 %424, label %425, label %436

425:                                              ; preds = %422
  %426 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %84, 1
  %427 = getelementptr inbounds nuw i32, ptr %426, i64 %423
  %428 = load i32, ptr %427, align 4
  %429 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %95, 1
  %430 = getelementptr inbounds nuw i32, ptr %429, i64 %423
  %431 = load i32, ptr %430, align 4
  %432 = add i32 %428, %431
  %433 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %84, 1
  %434 = getelementptr inbounds nuw i32, ptr %433, i64 %423
  store i32 %432, ptr %434, align 4
  %435 = add i64 %423, 1
  br label %422

436:                                              ; preds = %422
  %437 = call ptr @malloc(i64 576)
  %438 = ptrtoint ptr %437 to i64
  %439 = add i64 %438, 63
  %440 = urem i64 %439, 64
  %441 = sub i64 %439, %440
  %442 = inttoptr i64 %441 to ptr
  %443 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %437, 0
  %444 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %443, ptr %442, 1
  %445 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %444, i64 0, 2
  %446 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %445, i64 512, 3, 0
  %447 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %446, i64 1, 4, 0
  br label %448

448:                                              ; preds = %451, %436
  %449 = phi i64 [ %461, %451 ], [ 0, %436 ]
  %450 = icmp slt i64 %449, 509
  br i1 %450, label %451, label %462

451:                                              ; preds = %448
  %452 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %84, 1
  %453 = getelementptr i32, ptr %452, i64 %449
  %454 = load <4 x i32>, ptr %453, align 4
  %455 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %115, 1
  %456 = getelementptr i32, ptr %455, i64 %449
  %457 = load <4 x i32>, ptr %456, align 4
  %458 = icmp slt <4 x i32> %454, %457
  %459 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %447, 1
  %460 = getelementptr i1, ptr %459, i64 %449
  store <4 x i1> %458, ptr %460, align 1
  %461 = add i64 %449, 4
  br label %448

462:                                              ; preds = %448
  br label %463

463:                                              ; preds = %466, %462
  %464 = phi i64 [ %476, %466 ], [ 512, %462 ]
  %465 = icmp slt i64 %464, 512
  br i1 %465, label %466, label %477

466:                                              ; preds = %463
  %467 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %84, 1
  %468 = getelementptr inbounds nuw i32, ptr %467, i64 %464
  %469 = load i32, ptr %468, align 4
  %470 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %115, 1
  %471 = getelementptr inbounds nuw i32, ptr %470, i64 %464
  %472 = load i32, ptr %471, align 4
  %473 = icmp slt i32 %469, %472
  %474 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %447, 1
  %475 = getelementptr inbounds nuw i1, ptr %474, i64 %464
  store i1 %473, ptr %475, align 1
  %476 = add i64 %464, 1
  br label %463

477:                                              ; preds = %463
  %478 = sext i32 %385 to i64
  %479 = add i64 %73, %478
  %480 = extractvalue { i64, ptr } %33, 1
  %481 = load ptr, ptr %480, align 8
  %482 = getelementptr ptr, ptr %480, i32 1
  %483 = load ptr, ptr %482, align 8
  %484 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %481, 0
  %485 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %484, ptr %483, 1
  %486 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %485, i64 %479, 2
  %487 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %486, i64 512, 3, 0
  %488 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %487, i64 1, 4, 0
  %489 = add i64 %478, 512
  %490 = sext i32 %13 to i64
  %491 = call i64 @llvm.smin.i64(i64 %489, i64 %490)
  %492 = call i64 @llvm.smax.i64(i64 %491, i64 %478)
  %493 = sub i64 %492, %478
  %494 = call ptr @malloc(i64 1024)
  %495 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %494, 0
  %496 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %495, ptr %494, 1
  %497 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %496, i64 0, 2
  %498 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %497, i64 512, 3, 0
  %499 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %498, i64 1, 4, 0
  %500 = icmp slt i64 %493, 512
  br i1 %500, label %501, label %518

501:                                              ; preds = %477
  br label %502

502:                                              ; preds = %505, %501
  %503 = phi i64 [ %508, %505 ], [ 0, %501 ]
  %504 = icmp slt i64 %503, 509
  br i1 %504, label %505, label %509

505:                                              ; preds = %502
  %506 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %499, 1
  %507 = getelementptr half, ptr %506, i64 %503
  store <4 x half> zeroinitializer, ptr %507, align 2
  %508 = add i64 %503, 4
  br label %502

509:                                              ; preds = %502
  br label %510

510:                                              ; preds = %513, %509
  %511 = phi i64 [ %516, %513 ], [ 512, %509 ]
  %512 = icmp slt i64 %511, 512
  br i1 %512, label %513, label %517

513:                                              ; preds = %510
  %514 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %499, 1
  %515 = getelementptr inbounds nuw half, ptr %514, i64 %511
  store half 0xH0000, ptr %515, align 2
  %516 = add i64 %511, 1
  br label %510

517:                                              ; preds = %510
  br label %518

518:                                              ; preds = %517, %477
  %519 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %488, 0
  %520 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %488, 1
  %521 = insertvalue { ptr, ptr, i64 } poison, ptr %519, 0
  %522 = insertvalue { ptr, ptr, i64 } %521, ptr %520, 1
  %523 = insertvalue { ptr, ptr, i64 } %522, i64 0, 2
  %524 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %488, 2
  %525 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %488, 3, 0
  %526 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %488, 4, 0
  %527 = extractvalue { ptr, ptr, i64 } %523, 0
  %528 = extractvalue { ptr, ptr, i64 } %523, 1
  %529 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %527, 0
  %530 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %529, ptr %528, 1
  %531 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %530, i64 %524, 2
  %532 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %531, i64 %493, 3, 0
  %533 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %532, i64 1, 4, 0
  %534 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %499, 0
  %535 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %499, 1
  %536 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %534, 0
  %537 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %536, ptr %535, 1
  %538 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %537, i64 0, 2
  %539 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %538, i64 %493, 3, 0
  %540 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %539, i64 1, 4, 0
  %541 = call ptr @llvm.stacksave.p0()
  %542 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %533, ptr %542, align 8
  %543 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %542, 1
  %544 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %540, ptr %544, align 8
  %545 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %544, 1
  %546 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %543, ptr %546, align 8
  %547 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %545, ptr %547, align 8
  call void @memrefCopy(i64 2, ptr %546, ptr %547)
  call void @llvm.stackrestore.p0(ptr %541)
  br label %548

548:                                              ; preds = %551, %518
  %549 = phi i64 [ %558, %551 ], [ 0, %518 ]
  %550 = icmp slt i64 %549, 509
  br i1 %550, label %551, label %559

551:                                              ; preds = %548
  %552 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %499, 1
  %553 = getelementptr half, ptr %552, i64 %549
  %554 = load <4 x half>, ptr %553, align 2
  %555 = fpext <4 x half> %554 to <4 x float>
  %556 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %557 = getelementptr float, ptr %556, i64 %549
  store <4 x float> %555, ptr %557, align 4
  %558 = add i64 %549, 4
  br label %548

559:                                              ; preds = %548
  br label %560

560:                                              ; preds = %563, %559
  %561 = phi i64 [ %570, %563 ], [ 512, %559 ]
  %562 = icmp slt i64 %561, 512
  br i1 %562, label %563, label %571

563:                                              ; preds = %560
  %564 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %499, 1
  %565 = getelementptr inbounds nuw half, ptr %564, i64 %561
  %566 = load half, ptr %565, align 2
  %567 = fpext half %566 to float
  %568 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %569 = getelementptr inbounds nuw float, ptr %568, i64 %561
  store float %567, ptr %569, align 4
  %570 = add i64 %561, 1
  br label %560

571:                                              ; preds = %560
  br label %572

572:                                              ; preds = %575, %571
  %573 = phi i64 [ %585, %575 ], [ 0, %571 ]
  %574 = icmp slt i64 %573, 509
  br i1 %574, label %575, label %586

575:                                              ; preds = %572
  %576 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %577 = getelementptr float, ptr %576, i64 %573
  %578 = load <4 x float>, ptr %577, align 4
  %579 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %345, 1
  %580 = getelementptr float, ptr %579, i64 %573
  %581 = load <4 x float>, ptr %580, align 4
  %582 = fsub <4 x float> %578, %581
  %583 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %584 = getelementptr float, ptr %583, i64 %573
  store <4 x float> %582, ptr %584, align 4
  %585 = add i64 %573, 4
  br label %572

586:                                              ; preds = %572
  br label %587

587:                                              ; preds = %590, %586
  %588 = phi i64 [ %600, %590 ], [ 512, %586 ]
  %589 = icmp slt i64 %588, 512
  br i1 %589, label %590, label %601

590:                                              ; preds = %587
  %591 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %592 = getelementptr inbounds nuw float, ptr %591, i64 %588
  %593 = load float, ptr %592, align 4
  %594 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %345, 1
  %595 = getelementptr inbounds nuw float, ptr %594, i64 %588
  %596 = load float, ptr %595, align 4
  %597 = fsub float %593, %596
  %598 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %599 = getelementptr inbounds nuw float, ptr %598, i64 %588
  store float %597, ptr %599, align 4
  %600 = add i64 %588, 1
  br label %587

601:                                              ; preds = %587
  br label %602

602:                                              ; preds = %605, %601
  %603 = phi i64 [ %618, %605 ], [ 0, %601 ]
  %604 = icmp slt i64 %603, 509
  br i1 %604, label %605, label %619

605:                                              ; preds = %602
  %606 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %447, 1
  %607 = getelementptr i1, ptr %606, i64 %603
  %608 = load <4 x i1>, ptr %607, align 1
  %609 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %610 = getelementptr float, ptr %609, i64 %603
  %611 = load <4 x float>, ptr %610, align 4
  %612 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 1
  %613 = getelementptr float, ptr %612, i64 %603
  %614 = load <4 x float>, ptr %613, align 4
  %615 = select <4 x i1> %608, <4 x float> %611, <4 x float> %614
  %616 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %617 = getelementptr float, ptr %616, i64 %603
  store <4 x float> %615, ptr %617, align 4
  %618 = add i64 %603, 4
  br label %602

619:                                              ; preds = %602
  br label %620

620:                                              ; preds = %623, %619
  %621 = phi i64 [ %636, %623 ], [ 512, %619 ]
  %622 = icmp slt i64 %621, 512
  br i1 %622, label %623, label %637

623:                                              ; preds = %620
  %624 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %447, 1
  %625 = getelementptr inbounds nuw i1, ptr %624, i64 %621
  %626 = load i1, ptr %625, align 1
  %627 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %628 = getelementptr inbounds nuw float, ptr %627, i64 %621
  %629 = load float, ptr %628, align 4
  %630 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 1
  %631 = getelementptr inbounds nuw float, ptr %630, i64 %621
  %632 = load float, ptr %631, align 4
  %633 = select i1 %626, float %629, float %632
  %634 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %635 = getelementptr inbounds nuw float, ptr %634, i64 %621
  store float %633, ptr %635, align 4
  %636 = add i64 %621, 1
  br label %620

637:                                              ; preds = %620
  br label %638

638:                                              ; preds = %641, %637
  %639 = phi i64 [ %648, %641 ], [ 0, %637 ]
  %640 = icmp slt i64 %639, 509
  br i1 %640, label %641, label %649

641:                                              ; preds = %638
  %642 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %643 = getelementptr float, ptr %642, i64 %639
  %644 = load <4 x float>, ptr %643, align 4
  %645 = fmul <4 x float> %644, %644
  %646 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %647 = getelementptr float, ptr %646, i64 %639
  store <4 x float> %645, ptr %647, align 4
  %648 = add i64 %639, 4
  br label %638

649:                                              ; preds = %638
  br label %650

650:                                              ; preds = %653, %649
  %651 = phi i64 [ %660, %653 ], [ 512, %649 ]
  %652 = icmp slt i64 %651, 512
  br i1 %652, label %653, label %661

653:                                              ; preds = %650
  %654 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %655 = getelementptr inbounds nuw float, ptr %654, i64 %651
  %656 = load float, ptr %655, align 4
  %657 = fmul float %656, %656
  %658 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %659 = getelementptr inbounds nuw float, ptr %658, i64 %651
  store float %657, ptr %659, align 4
  %660 = add i64 %651, 1
  br label %650

661:                                              ; preds = %650
  br label %662

662:                                              ; preds = %665, %661
  %663 = phi i64 [ %675, %665 ], [ 0, %661 ]
  %664 = icmp slt i64 %663, 509
  br i1 %664, label %665, label %676

665:                                              ; preds = %662
  %666 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %386, 1
  %667 = getelementptr float, ptr %666, i64 %663
  %668 = load <4 x float>, ptr %667, align 4
  %669 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %670 = getelementptr float, ptr %669, i64 %663
  %671 = load <4 x float>, ptr %670, align 4
  %672 = fadd <4 x float> %668, %671
  %673 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %386, 1
  %674 = getelementptr float, ptr %673, i64 %663
  store <4 x float> %672, ptr %674, align 4
  %675 = add i64 %663, 4
  br label %662

676:                                              ; preds = %662
  br label %677

677:                                              ; preds = %680, %676
  %678 = phi i64 [ %690, %680 ], [ 512, %676 ]
  %679 = icmp slt i64 %678, 512
  br i1 %679, label %680, label %691

680:                                              ; preds = %677
  %681 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %386, 1
  %682 = getelementptr inbounds nuw float, ptr %681, i64 %678
  %683 = load float, ptr %682, align 4
  %684 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %685 = getelementptr inbounds nuw float, ptr %684, i64 %678
  %686 = load float, ptr %685, align 4
  %687 = fadd float %683, %686
  %688 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %386, 1
  %689 = getelementptr inbounds nuw float, ptr %688, i64 %678
  store float %687, ptr %689, align 4
  %690 = add i64 %678, 1
  br label %677

691:                                              ; preds = %677
  %692 = add i32 %385, 512
  br label %384

693:                                              ; preds = %384
  %694 = call ptr @malloc(i64 68)
  %695 = ptrtoint ptr %694 to i64
  %696 = add i64 %695, 63
  %697 = urem i64 %696, 64
  %698 = sub i64 %696, %697
  %699 = inttoptr i64 %698 to ptr
  %700 = insertvalue { ptr, ptr, i64 } poison, ptr %694, 0
  %701 = insertvalue { ptr, ptr, i64 } %700, ptr %699, 1
  %702 = insertvalue { ptr, ptr, i64 } %701, i64 0, 2
  %703 = extractvalue { ptr, ptr, i64 } %702, 1
  store float 0.000000e+00, ptr %703, align 4
  %704 = alloca float, i64 1, align 4
  %705 = insertvalue { ptr, ptr, i64 } poison, ptr %704, 0
  %706 = insertvalue { ptr, ptr, i64 } %705, ptr %704, 1
  %707 = insertvalue { ptr, ptr, i64 } %706, i64 0, 2
  %708 = extractvalue { ptr, ptr, i64 } %702, 1
  %709 = load float, ptr %708, align 4
  %710 = extractvalue { ptr, ptr, i64 } %707, 1
  store float %709, ptr %710, align 4
  br label %711

711:                                              ; preds = %714, %693
  %712 = phi i64 [ %722, %714 ], [ 0, %693 ]
  %713 = icmp slt i64 %712, 509
  br i1 %713, label %714, label %723

714:                                              ; preds = %711
  %715 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %386, 1
  %716 = getelementptr float, ptr %715, i64 %712
  %717 = load <4 x float>, ptr %716, align 4
  %718 = extractvalue { ptr, ptr, i64 } %707, 1
  %719 = load float, ptr %718, align 4
  %720 = call float @llvm.vector.reduce.fadd.v4f32(float %719, <4 x float> %717)
  %721 = extractvalue { ptr, ptr, i64 } %707, 1
  store float %720, ptr %721, align 4
  %722 = add i64 %712, 4
  br label %711

723:                                              ; preds = %711
  br label %724

724:                                              ; preds = %727, %723
  %725 = phi i64 [ %735, %727 ], [ 512, %723 ]
  %726 = icmp slt i64 %725, 512
  br i1 %726, label %727, label %736

727:                                              ; preds = %724
  %728 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %386, 1
  %729 = getelementptr inbounds nuw float, ptr %728, i64 %725
  %730 = load float, ptr %729, align 4
  %731 = extractvalue { ptr, ptr, i64 } %707, 1
  %732 = load float, ptr %731, align 4
  %733 = fadd float %730, %732
  %734 = extractvalue { ptr, ptr, i64 } %707, 1
  store float %733, ptr %734, align 4
  %735 = add i64 %725, 1
  br label %724

736:                                              ; preds = %724
  %737 = extractvalue { ptr, ptr, i64 } %707, 1
  %738 = load float, ptr %737, align 4
  %739 = extractvalue { ptr, ptr, i64 } %702, 1
  store float %738, ptr %739, align 4
  %740 = extractvalue { ptr, ptr, i64 } %702, 1
  %741 = load float, ptr %740, align 4
  %742 = fdiv float %741, %333
  %743 = fadd float %742, %14
  %744 = call float @llvm.sqrt.f32(float %743)
  %745 = fdiv float 1.000000e+00, %744
  %746 = sext i32 %18 to i64
  %747 = extractvalue { i64, ptr } %25, 1
  %748 = load ptr, ptr %747, align 8
  %749 = getelementptr ptr, ptr %747, i32 1
  %750 = load ptr, ptr %749, align 8
  %751 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %748, 0
  %752 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %751, ptr %750, 1
  %753 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %752, i64 %746, 2
  %754 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %753, i64 1, 3, 0
  %755 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %754, i64 1, 4, 0
  %756 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %755, 1
  %757 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %755, 2
  %758 = getelementptr float, ptr %756, i64 %757
  %759 = getelementptr inbounds nuw float, ptr %758, i64 0
  store float %334, ptr %759, align 4
  %760 = extractvalue { i64, ptr } %23, 1
  %761 = load ptr, ptr %760, align 8
  %762 = getelementptr ptr, ptr %760, i32 1
  %763 = load ptr, ptr %762, align 8
  %764 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %761, 0
  %765 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %764, ptr %763, 1
  %766 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %765, i64 %746, 2
  %767 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %766, i64 1, 3, 0
  %768 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %767, i64 1, 4, 0
  %769 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %768, 1
  %770 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %768, 2
  %771 = getelementptr float, ptr %769, i64 %770
  %772 = getelementptr inbounds nuw float, ptr %771, i64 0
  store float %745, ptr %772, align 4
  %773 = call ptr @malloc(i64 2112)
  %774 = ptrtoint ptr %773 to i64
  %775 = add i64 %774, 63
  %776 = urem i64 %775, 64
  %777 = sub i64 %775, %776
  %778 = inttoptr i64 %777 to ptr
  %779 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %773, 0
  %780 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %779, ptr %778, 1
  %781 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %780, i64 0, 2
  %782 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %781, i64 512, 3, 0
  %783 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %782, i64 1, 4, 0
  br label %784

784:                                              ; preds = %787, %736
  %785 = phi i64 [ %792, %787 ], [ 0, %736 ]
  %786 = icmp slt i64 %785, 509
  br i1 %786, label %787, label %793

787:                                              ; preds = %784
  %788 = insertelement <4 x float> poison, float %745, i32 0
  %789 = shufflevector <4 x float> %788, <4 x float> poison, <4 x i32> zeroinitializer
  %790 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %783, 1
  %791 = getelementptr float, ptr %790, i64 %785
  store <4 x float> %789, ptr %791, align 4
  %792 = add i64 %785, 4
  br label %784

793:                                              ; preds = %784
  br label %794

794:                                              ; preds = %797, %793
  %795 = phi i64 [ %800, %797 ], [ 512, %793 ]
  %796 = icmp slt i64 %795, 512
  br i1 %796, label %797, label %801

797:                                              ; preds = %794
  %798 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %783, 1
  %799 = getelementptr inbounds nuw float, ptr %798, i64 %795
  store float %745, ptr %799, align 4
  %800 = add i64 %795, 1
  br label %794

801:                                              ; preds = %794
  br label %802

802:                                              ; preds = %1210, %801
  %803 = phi i32 [ %1240, %1210 ], [ 0, %801 ]
  %804 = icmp slt i32 %803, %13
  br i1 %804, label %805, label %1241

805:                                              ; preds = %802
  %806 = sext i32 %803 to i64
  %807 = extractvalue { i64, ptr } %29, 1
  %808 = load ptr, ptr %807, align 8
  %809 = getelementptr ptr, ptr %807, i32 1
  %810 = load ptr, ptr %809, align 8
  %811 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %808, 0
  %812 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %811, ptr %810, 1
  %813 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %812, i64 %806, 2
  %814 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %813, i64 512, 3, 0
  %815 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %814, i64 1, 4, 0
  %816 = add i64 %806, 512
  %817 = sext i32 %13 to i64
  %818 = call i64 @llvm.smin.i64(i64 %816, i64 %817)
  %819 = call i64 @llvm.smax.i64(i64 %818, i64 %806)
  %820 = sub i64 %819, %806
  %821 = call ptr @malloc(i64 1024)
  %822 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %821, 0
  %823 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %822, ptr %821, 1
  %824 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %823, i64 0, 2
  %825 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %824, i64 512, 3, 0
  %826 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %825, i64 1, 4, 0
  %827 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %815, 0
  %828 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %815, 1
  %829 = insertvalue { ptr, ptr, i64 } poison, ptr %827, 0
  %830 = insertvalue { ptr, ptr, i64 } %829, ptr %828, 1
  %831 = insertvalue { ptr, ptr, i64 } %830, i64 0, 2
  %832 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %815, 2
  %833 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %815, 3, 0
  %834 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %815, 4, 0
  %835 = extractvalue { ptr, ptr, i64 } %831, 0
  %836 = extractvalue { ptr, ptr, i64 } %831, 1
  %837 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %835, 0
  %838 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %837, ptr %836, 1
  %839 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %838, i64 %832, 2
  %840 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %839, i64 %820, 3, 0
  %841 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %840, i64 1, 4, 0
  %842 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %826, 0
  %843 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %826, 1
  %844 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %842, 0
  %845 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %844, ptr %843, 1
  %846 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %845, i64 0, 2
  %847 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %846, i64 %820, 3, 0
  %848 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %847, i64 1, 4, 0
  %849 = call ptr @llvm.stacksave.p0()
  %850 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %841, ptr %850, align 8
  %851 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %850, 1
  %852 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %848, ptr %852, align 8
  %853 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %852, 1
  %854 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %851, ptr %854, align 8
  %855 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %853, ptr %855, align 8
  call void @memrefCopy(i64 2, ptr %854, ptr %855)
  call void @llvm.stackrestore.p0(ptr %849)
  %856 = extractvalue { i64, ptr } %27, 1
  %857 = load ptr, ptr %856, align 8
  %858 = getelementptr ptr, ptr %856, i32 1
  %859 = load ptr, ptr %858, align 8
  %860 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %857, 0
  %861 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %860, ptr %859, 1
  %862 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %861, i64 %806, 2
  %863 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %862, i64 512, 3, 0
  %864 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %863, i64 1, 4, 0
  %865 = call ptr @malloc(i64 1024)
  %866 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %865, 0
  %867 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %866, ptr %865, 1
  %868 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %867, i64 0, 2
  %869 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %868, i64 512, 3, 0
  %870 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %869, i64 1, 4, 0
  %871 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %864, 0
  %872 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %864, 1
  %873 = insertvalue { ptr, ptr, i64 } poison, ptr %871, 0
  %874 = insertvalue { ptr, ptr, i64 } %873, ptr %872, 1
  %875 = insertvalue { ptr, ptr, i64 } %874, i64 0, 2
  %876 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %864, 2
  %877 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %864, 3, 0
  %878 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %864, 4, 0
  %879 = extractvalue { ptr, ptr, i64 } %875, 0
  %880 = extractvalue { ptr, ptr, i64 } %875, 1
  %881 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %879, 0
  %882 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %881, ptr %880, 1
  %883 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %882, i64 %876, 2
  %884 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %883, i64 %820, 3, 0
  %885 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %884, i64 1, 4, 0
  %886 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %870, 0
  %887 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %870, 1
  %888 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %886, 0
  %889 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %888, ptr %887, 1
  %890 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %889, i64 0, 2
  %891 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %890, i64 %820, 3, 0
  %892 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %891, i64 1, 4, 0
  %893 = call ptr @llvm.stacksave.p0()
  %894 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %885, ptr %894, align 8
  %895 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %894, 1
  %896 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %892, ptr %896, align 8
  %897 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %896, 1
  %898 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %895, ptr %898, align 8
  %899 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %897, ptr %899, align 8
  call void @memrefCopy(i64 2, ptr %898, ptr %899)
  call void @llvm.stackrestore.p0(ptr %893)
  %900 = add i64 %73, %806
  %901 = extractvalue { i64, ptr } %33, 1
  %902 = load ptr, ptr %901, align 8
  %903 = getelementptr ptr, ptr %901, i32 1
  %904 = load ptr, ptr %903, align 8
  %905 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %902, 0
  %906 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %905, ptr %904, 1
  %907 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %906, i64 %900, 2
  %908 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %907, i64 512, 3, 0
  %909 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %908, i64 1, 4, 0
  %910 = call ptr @malloc(i64 1024)
  %911 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %910, 0
  %912 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %911, ptr %910, 1
  %913 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %912, i64 0, 2
  %914 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %913, i64 512, 3, 0
  %915 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %914, i64 1, 4, 0
  %916 = icmp slt i64 %820, 512
  br i1 %916, label %917, label %934

917:                                              ; preds = %805
  br label %918

918:                                              ; preds = %921, %917
  %919 = phi i64 [ %924, %921 ], [ 0, %917 ]
  %920 = icmp slt i64 %919, 509
  br i1 %920, label %921, label %925

921:                                              ; preds = %918
  %922 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %915, 1
  %923 = getelementptr half, ptr %922, i64 %919
  store <4 x half> zeroinitializer, ptr %923, align 2
  %924 = add i64 %919, 4
  br label %918

925:                                              ; preds = %918
  br label %926

926:                                              ; preds = %929, %925
  %927 = phi i64 [ %932, %929 ], [ 512, %925 ]
  %928 = icmp slt i64 %927, 512
  br i1 %928, label %929, label %933

929:                                              ; preds = %926
  %930 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %915, 1
  %931 = getelementptr inbounds nuw half, ptr %930, i64 %927
  store half 0xH0000, ptr %931, align 2
  %932 = add i64 %927, 1
  br label %926

933:                                              ; preds = %926
  br label %934

934:                                              ; preds = %933, %805
  %935 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %909, 0
  %936 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %909, 1
  %937 = insertvalue { ptr, ptr, i64 } poison, ptr %935, 0
  %938 = insertvalue { ptr, ptr, i64 } %937, ptr %936, 1
  %939 = insertvalue { ptr, ptr, i64 } %938, i64 0, 2
  %940 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %909, 2
  %941 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %909, 3, 0
  %942 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %909, 4, 0
  %943 = extractvalue { ptr, ptr, i64 } %939, 0
  %944 = extractvalue { ptr, ptr, i64 } %939, 1
  %945 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %943, 0
  %946 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %945, ptr %944, 1
  %947 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %946, i64 %940, 2
  %948 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %947, i64 %820, 3, 0
  %949 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %948, i64 1, 4, 0
  %950 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %915, 0
  %951 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %915, 1
  %952 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %950, 0
  %953 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %952, ptr %951, 1
  %954 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %953, i64 0, 2
  %955 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %954, i64 %820, 3, 0
  %956 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %955, i64 1, 4, 0
  %957 = call ptr @llvm.stacksave.p0()
  %958 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %949, ptr %958, align 8
  %959 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %958, 1
  %960 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %956, ptr %960, align 8
  %961 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %960, 1
  %962 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %959, ptr %962, align 8
  %963 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %961, ptr %963, align 8
  call void @memrefCopy(i64 2, ptr %962, ptr %963)
  call void @llvm.stackrestore.p0(ptr %957)
  %964 = call ptr @malloc(i64 2112)
  %965 = ptrtoint ptr %964 to i64
  %966 = add i64 %965, 63
  %967 = urem i64 %966, 64
  %968 = sub i64 %966, %967
  %969 = inttoptr i64 %968 to ptr
  %970 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %964, 0
  %971 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %970, ptr %969, 1
  %972 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %971, i64 0, 2
  %973 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %972, i64 512, 3, 0
  %974 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %973, i64 1, 4, 0
  br label %975

975:                                              ; preds = %978, %934
  %976 = phi i64 [ %985, %978 ], [ 0, %934 ]
  %977 = icmp slt i64 %976, 509
  br i1 %977, label %978, label %986

978:                                              ; preds = %975
  %979 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %915, 1
  %980 = getelementptr half, ptr %979, i64 %976
  %981 = load <4 x half>, ptr %980, align 2
  %982 = fpext <4 x half> %981 to <4 x float>
  %983 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %974, 1
  %984 = getelementptr float, ptr %983, i64 %976
  store <4 x float> %982, ptr %984, align 4
  %985 = add i64 %976, 4
  br label %975

986:                                              ; preds = %975
  br label %987

987:                                              ; preds = %990, %986
  %988 = phi i64 [ %997, %990 ], [ 512, %986 ]
  %989 = icmp slt i64 %988, 512
  br i1 %989, label %990, label %998

990:                                              ; preds = %987
  %991 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %915, 1
  %992 = getelementptr inbounds nuw half, ptr %991, i64 %988
  %993 = load half, ptr %992, align 2
  %994 = fpext half %993 to float
  %995 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %974, 1
  %996 = getelementptr inbounds nuw float, ptr %995, i64 %988
  store float %994, ptr %996, align 4
  %997 = add i64 %988, 1
  br label %987

998:                                              ; preds = %987
  br label %999

999:                                              ; preds = %1002, %998
  %1000 = phi i64 [ %1012, %1002 ], [ 0, %998 ]
  %1001 = icmp slt i64 %1000, 509
  br i1 %1001, label %1002, label %1013

1002:                                             ; preds = %999
  %1003 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %974, 1
  %1004 = getelementptr float, ptr %1003, i64 %1000
  %1005 = load <4 x float>, ptr %1004, align 4
  %1006 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %345, 1
  %1007 = getelementptr float, ptr %1006, i64 %1000
  %1008 = load <4 x float>, ptr %1007, align 4
  %1009 = fsub <4 x float> %1005, %1008
  %1010 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %974, 1
  %1011 = getelementptr float, ptr %1010, i64 %1000
  store <4 x float> %1009, ptr %1011, align 4
  %1012 = add i64 %1000, 4
  br label %999

1013:                                             ; preds = %999
  br label %1014

1014:                                             ; preds = %1017, %1013
  %1015 = phi i64 [ %1027, %1017 ], [ 512, %1013 ]
  %1016 = icmp slt i64 %1015, 512
  br i1 %1016, label %1017, label %1028

1017:                                             ; preds = %1014
  %1018 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %974, 1
  %1019 = getelementptr inbounds nuw float, ptr %1018, i64 %1015
  %1020 = load float, ptr %1019, align 4
  %1021 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %345, 1
  %1022 = getelementptr inbounds nuw float, ptr %1021, i64 %1015
  %1023 = load float, ptr %1022, align 4
  %1024 = fsub float %1020, %1023
  %1025 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %974, 1
  %1026 = getelementptr inbounds nuw float, ptr %1025, i64 %1015
  store float %1024, ptr %1026, align 4
  %1027 = add i64 %1015, 1
  br label %1014

1028:                                             ; preds = %1014
  br label %1029

1029:                                             ; preds = %1032, %1028
  %1030 = phi i64 [ %1042, %1032 ], [ 0, %1028 ]
  %1031 = icmp slt i64 %1030, 509
  br i1 %1031, label %1032, label %1043

1032:                                             ; preds = %1029
  %1033 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %974, 1
  %1034 = getelementptr float, ptr %1033, i64 %1030
  %1035 = load <4 x float>, ptr %1034, align 4
  %1036 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %783, 1
  %1037 = getelementptr float, ptr %1036, i64 %1030
  %1038 = load <4 x float>, ptr %1037, align 4
  %1039 = fmul <4 x float> %1035, %1038
  %1040 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %974, 1
  %1041 = getelementptr float, ptr %1040, i64 %1030
  store <4 x float> %1039, ptr %1041, align 4
  %1042 = add i64 %1030, 4
  br label %1029

1043:                                             ; preds = %1029
  br label %1044

1044:                                             ; preds = %1047, %1043
  %1045 = phi i64 [ %1057, %1047 ], [ 512, %1043 ]
  %1046 = icmp slt i64 %1045, 512
  br i1 %1046, label %1047, label %1058

1047:                                             ; preds = %1044
  %1048 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %974, 1
  %1049 = getelementptr inbounds nuw float, ptr %1048, i64 %1045
  %1050 = load float, ptr %1049, align 4
  %1051 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %783, 1
  %1052 = getelementptr inbounds nuw float, ptr %1051, i64 %1045
  %1053 = load float, ptr %1052, align 4
  %1054 = fmul float %1050, %1053
  %1055 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %974, 1
  %1056 = getelementptr inbounds nuw float, ptr %1055, i64 %1045
  store float %1054, ptr %1056, align 4
  %1057 = add i64 %1045, 1
  br label %1044

1058:                                             ; preds = %1044
  br label %1059

1059:                                             ; preds = %1062, %1058
  %1060 = phi i64 [ %1069, %1062 ], [ 0, %1058 ]
  %1061 = icmp slt i64 %1060, 509
  br i1 %1061, label %1062, label %1070

1062:                                             ; preds = %1059
  %1063 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %826, 1
  %1064 = getelementptr half, ptr %1063, i64 %1060
  %1065 = load <4 x half>, ptr %1064, align 2
  %1066 = fpext <4 x half> %1065 to <4 x float>
  %1067 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %1068 = getelementptr float, ptr %1067, i64 %1060
  store <4 x float> %1066, ptr %1068, align 4
  %1069 = add i64 %1060, 4
  br label %1059

1070:                                             ; preds = %1059
  br label %1071

1071:                                             ; preds = %1074, %1070
  %1072 = phi i64 [ %1081, %1074 ], [ 512, %1070 ]
  %1073 = icmp slt i64 %1072, 512
  br i1 %1073, label %1074, label %1082

1074:                                             ; preds = %1071
  %1075 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %826, 1
  %1076 = getelementptr inbounds nuw half, ptr %1075, i64 %1072
  %1077 = load half, ptr %1076, align 2
  %1078 = fpext half %1077 to float
  %1079 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %1080 = getelementptr inbounds nuw float, ptr %1079, i64 %1072
  store float %1078, ptr %1080, align 4
  %1081 = add i64 %1072, 1
  br label %1071

1082:                                             ; preds = %1071
  br label %1083

1083:                                             ; preds = %1086, %1082
  %1084 = phi i64 [ %1096, %1086 ], [ 0, %1082 ]
  %1085 = icmp slt i64 %1084, 509
  br i1 %1085, label %1086, label %1097

1086:                                             ; preds = %1083
  %1087 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %974, 1
  %1088 = getelementptr float, ptr %1087, i64 %1084
  %1089 = load <4 x float>, ptr %1088, align 4
  %1090 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %1091 = getelementptr float, ptr %1090, i64 %1084
  %1092 = load <4 x float>, ptr %1091, align 4
  %1093 = fmul <4 x float> %1089, %1092
  %1094 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %974, 1
  %1095 = getelementptr float, ptr %1094, i64 %1084
  store <4 x float> %1093, ptr %1095, align 4
  %1096 = add i64 %1084, 4
  br label %1083

1097:                                             ; preds = %1083
  br label %1098

1098:                                             ; preds = %1101, %1097
  %1099 = phi i64 [ %1111, %1101 ], [ 512, %1097 ]
  %1100 = icmp slt i64 %1099, 512
  br i1 %1100, label %1101, label %1112

1101:                                             ; preds = %1098
  %1102 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %974, 1
  %1103 = getelementptr inbounds nuw float, ptr %1102, i64 %1099
  %1104 = load float, ptr %1103, align 4
  %1105 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %1106 = getelementptr inbounds nuw float, ptr %1105, i64 %1099
  %1107 = load float, ptr %1106, align 4
  %1108 = fmul float %1104, %1107
  %1109 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %974, 1
  %1110 = getelementptr inbounds nuw float, ptr %1109, i64 %1099
  store float %1108, ptr %1110, align 4
  %1111 = add i64 %1099, 1
  br label %1098

1112:                                             ; preds = %1098
  br label %1113

1113:                                             ; preds = %1116, %1112
  %1114 = phi i64 [ %1123, %1116 ], [ 0, %1112 ]
  %1115 = icmp slt i64 %1114, 509
  br i1 %1115, label %1116, label %1124

1116:                                             ; preds = %1113
  %1117 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %870, 1
  %1118 = getelementptr half, ptr %1117, i64 %1114
  %1119 = load <4 x half>, ptr %1118, align 2
  %1120 = fpext <4 x half> %1119 to <4 x float>
  %1121 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %1122 = getelementptr float, ptr %1121, i64 %1114
  store <4 x float> %1120, ptr %1122, align 4
  %1123 = add i64 %1114, 4
  br label %1113

1124:                                             ; preds = %1113
  br label %1125

1125:                                             ; preds = %1128, %1124
  %1126 = phi i64 [ %1135, %1128 ], [ 512, %1124 ]
  %1127 = icmp slt i64 %1126, 512
  br i1 %1127, label %1128, label %1136

1128:                                             ; preds = %1125
  %1129 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %870, 1
  %1130 = getelementptr inbounds nuw half, ptr %1129, i64 %1126
  %1131 = load half, ptr %1130, align 2
  %1132 = fpext half %1131 to float
  %1133 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %1134 = getelementptr inbounds nuw float, ptr %1133, i64 %1126
  store float %1132, ptr %1134, align 4
  %1135 = add i64 %1126, 1
  br label %1125

1136:                                             ; preds = %1125
  br label %1137

1137:                                             ; preds = %1140, %1136
  %1138 = phi i64 [ %1150, %1140 ], [ 0, %1136 ]
  %1139 = icmp slt i64 %1138, 509
  br i1 %1139, label %1140, label %1151

1140:                                             ; preds = %1137
  %1141 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %974, 1
  %1142 = getelementptr float, ptr %1141, i64 %1138
  %1143 = load <4 x float>, ptr %1142, align 4
  %1144 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %1145 = getelementptr float, ptr %1144, i64 %1138
  %1146 = load <4 x float>, ptr %1145, align 4
  %1147 = fadd <4 x float> %1143, %1146
  %1148 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %974, 1
  %1149 = getelementptr float, ptr %1148, i64 %1138
  store <4 x float> %1147, ptr %1149, align 4
  %1150 = add i64 %1138, 4
  br label %1137

1151:                                             ; preds = %1137
  br label %1152

1152:                                             ; preds = %1155, %1151
  %1153 = phi i64 [ %1165, %1155 ], [ 512, %1151 ]
  %1154 = icmp slt i64 %1153, 512
  br i1 %1154, label %1155, label %1166

1155:                                             ; preds = %1152
  %1156 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %974, 1
  %1157 = getelementptr inbounds nuw float, ptr %1156, i64 %1153
  %1158 = load float, ptr %1157, align 4
  %1159 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, 1
  %1160 = getelementptr inbounds nuw float, ptr %1159, i64 %1153
  %1161 = load float, ptr %1160, align 4
  %1162 = fadd float %1158, %1161
  %1163 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %974, 1
  %1164 = getelementptr inbounds nuw float, ptr %1163, i64 %1153
  store float %1162, ptr %1164, align 4
  %1165 = add i64 %1153, 1
  br label %1152

1166:                                             ; preds = %1152
  %1167 = extractvalue { i64, ptr } %31, 1
  %1168 = load ptr, ptr %1167, align 8
  %1169 = getelementptr ptr, ptr %1167, i32 1
  %1170 = load ptr, ptr %1169, align 8
  %1171 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %1168, 0
  %1172 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1171, ptr %1170, 1
  %1173 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1172, i64 %900, 2
  %1174 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1173, i64 512, 3, 0
  %1175 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1174, i64 1, 4, 0
  %1176 = call ptr @malloc(i64 1088)
  %1177 = ptrtoint ptr %1176 to i64
  %1178 = add i64 %1177, 63
  %1179 = urem i64 %1178, 64
  %1180 = sub i64 %1178, %1179
  %1181 = inttoptr i64 %1180 to ptr
  %1182 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %1176, 0
  %1183 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1182, ptr %1181, 1
  %1184 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1183, i64 0, 2
  %1185 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1184, i64 512, 3, 0
  %1186 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1185, i64 1, 4, 0
  br label %1187

1187:                                             ; preds = %1190, %1166
  %1188 = phi i64 [ %1197, %1190 ], [ 0, %1166 ]
  %1189 = icmp slt i64 %1188, 509
  br i1 %1189, label %1190, label %1198

1190:                                             ; preds = %1187
  %1191 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %974, 1
  %1192 = getelementptr float, ptr %1191, i64 %1188
  %1193 = load <4 x float>, ptr %1192, align 4
  %1194 = fptrunc <4 x float> %1193 to <4 x half>
  %1195 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1186, 1
  %1196 = getelementptr half, ptr %1195, i64 %1188
  store <4 x half> %1194, ptr %1196, align 2
  %1197 = add i64 %1188, 4
  br label %1187

1198:                                             ; preds = %1187
  br label %1199

1199:                                             ; preds = %1202, %1198
  %1200 = phi i64 [ %1209, %1202 ], [ 512, %1198 ]
  %1201 = icmp slt i64 %1200, 512
  br i1 %1201, label %1202, label %1210

1202:                                             ; preds = %1199
  %1203 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %974, 1
  %1204 = getelementptr inbounds nuw float, ptr %1203, i64 %1200
  %1205 = load float, ptr %1204, align 4
  %1206 = fptrunc float %1205 to half
  %1207 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1186, 1
  %1208 = getelementptr inbounds nuw half, ptr %1207, i64 %1200
  store half %1206, ptr %1208, align 2
  %1209 = add i64 %1200, 1
  br label %1199

1210:                                             ; preds = %1199
  %1211 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1186, 0
  %1212 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1186, 1
  %1213 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %1211, 0
  %1214 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1213, ptr %1212, 1
  %1215 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1214, i64 0, 2
  %1216 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1215, i64 %820, 3, 0
  %1217 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1216, i64 1, 4, 0
  %1218 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1175, 0
  %1219 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1175, 1
  %1220 = insertvalue { ptr, ptr, i64 } poison, ptr %1218, 0
  %1221 = insertvalue { ptr, ptr, i64 } %1220, ptr %1219, 1
  %1222 = insertvalue { ptr, ptr, i64 } %1221, i64 0, 2
  %1223 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1175, 2
  %1224 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1175, 3, 0
  %1225 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1175, 4, 0
  %1226 = extractvalue { ptr, ptr, i64 } %1222, 0
  %1227 = extractvalue { ptr, ptr, i64 } %1222, 1
  %1228 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %1226, 0
  %1229 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1228, ptr %1227, 1
  %1230 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1229, i64 %1223, 2
  %1231 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1230, i64 %820, 3, 0
  %1232 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1231, i64 1, 4, 0
  %1233 = call ptr @llvm.stacksave.p0()
  %1234 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %1217, ptr %1234, align 8
  %1235 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %1234, 1
  %1236 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %1232, ptr %1236, align 8
  %1237 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %1236, 1
  %1238 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %1235, ptr %1238, align 8
  %1239 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %1237, ptr %1239, align 8
  call void @memrefCopy(i64 2, ptr %1238, ptr %1239)
  call void @llvm.stackrestore.p0(ptr %1233)
  %1240 = add i32 %803, 512
  br label %802

1241:                                             ; preds = %802
  ret void
}

define i32 @main() {
  %1 = call ptr @malloc(i64 1024)
  %2 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %1, 0
  %3 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2, ptr %1, 1
  %4 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %3, i64 0, 2
  %5 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, i64 512, 3, 0
  %6 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, i64 1, 4, 0
  %7 = call ptr @malloc(i64 1024)
  %8 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %7, 0
  %9 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, ptr %7, 1
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, i64 0, 2
  %11 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, i64 512, 3, 0
  %12 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, i64 1, 4, 0
  %13 = call ptr @malloc(i64 1024)
  %14 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %13, 0
  %15 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, ptr %13, 1
  %16 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %15, i64 0, 2
  %17 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, i64 512, 3, 0
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, i64 1, 4, 0
  %19 = call ptr @malloc(i64 1024)
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %19, 0
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, ptr %19, 1
  %22 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, i64 0, 2
  %23 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, i64 512, 3, 0
  %24 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, i64 1, 4, 0
  %25 = call ptr @malloc(i64 4)
  %26 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %25, 0
  %27 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, ptr %25, 1
  %28 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %27, i64 0, 2
  %29 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, i64 1, 3, 0
  %30 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %29, i64 1, 4, 0
  %31 = call ptr @malloc(i64 4)
  %32 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %31, 0
  %33 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, ptr %31, 1
  %34 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, i64 0, 2
  %35 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %34, i64 1, 3, 0
  %36 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %35, i64 1, 4, 0
  %37 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, ptr %37, align 8
  %38 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %37, 1
  %39 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, ptr %39, align 8
  %40 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %39, 1
  %41 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, ptr %41, align 8
  %42 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %41, 1
  %43 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %24, ptr %43, align 8
  %44 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %43, 1
  %45 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %30, ptr %45, align 8
  %46 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %45, 1
  %47 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, ptr %47, align 8
  %48 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %47, 1
  %49 = extractvalue { i64, ptr } %38, 0
  %50 = extractvalue { i64, ptr } %38, 1
  %51 = extractvalue { i64, ptr } %40, 0
  %52 = extractvalue { i64, ptr } %40, 1
  %53 = extractvalue { i64, ptr } %42, 0
  %54 = extractvalue { i64, ptr } %42, 1
  %55 = extractvalue { i64, ptr } %44, 0
  %56 = extractvalue { i64, ptr } %44, 1
  %57 = extractvalue { i64, ptr } %46, 0
  %58 = extractvalue { i64, ptr } %46, 1
  %59 = extractvalue { i64, ptr } %48, 0
  %60 = extractvalue { i64, ptr } %48, 1
  call void @_layer_norm_fwd_fused(i64 %49, ptr %50, i64 %51, ptr %52, i64 %53, ptr %54, i64 %55, ptr %56, i64 %57, ptr %58, i64 %59, ptr %60, i32 512, i32 512, float 0x3EE4F8B580000000, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %61 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 0
  call void @free(ptr %61)
  %62 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 0
  call void @free(ptr %62)
  %63 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, 0
  call void @free(ptr %63)
  %64 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %24, 0
  call void @free(ptr %64)
  %65 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %30, 0
  call void @free(ptr %65)
  %66 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 0
  call void @free(ptr %66)
  ret i32 0
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #2

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.vector.reduce.fadd.v4f32(float, <4 x float>) #1

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
