#ifndef GEMMINI_TRANSLATE_H
#define GEMMINI_TRANSLATE_H
#define CONFIG_LD 1
#define CONFIG_ST 2
#define DIM 16
#define ADDR_LEN 32
#define ACC_SCALE_IDENTITY 1.0
#define MVIN_SCALE_IDENTITY 1.0
typedef uint32_t acc_scale_t_bits;
typedef float acc_scale_t;
typedef uint32_t scale_t_bits;
typedef float scale_t;

static acc_scale_t_bits acc_scale_t_to_acc_scale_t_bits(acc_scale_t x) {
    union {
        acc_scale_t_bits b;
        acc_scale_t f;
    } un;

    un.f = x;
    return un.b;
}

static scale_t_bits scale_t_to_scale_t_bits(scale_t x) {
    union {
        scale_t_bits b;
        scale_t f;
    } un;

    un.f = x;
    return un.b;
}

namespace mlir {

class LLVMConversionTarget;
class LLVMTypeConverter;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

void populateGemminiLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,RewritePatternSet &patterns);
void configureGemminiegalizeForExportTarget(LLVMConversionTarget &target);

}  // namespace mlir

#endif // GEMMINI_TRANSLATE_H