include "mlir/Pass/PassBase.td"

def LowerGemminiPass : Pass<"lower-gemmini", "mlir::ModuleOp"> {
  let summary = "lower Gemmini Dialect high level op to intrinsic op. ";
  let options = [
    Option<"dim", "dim", "int64_t", "16","Size of systolic array.">,
    Option<"addrLen", "addr_len", "int64_t", "32", "The length of address.">,
    Option<"accRows", "acc_rows", "int64_t", "1024", "The row of acc.">,
    Option<"bankRows", "bank_rows", "int64_t", "4096", "The row of the bank.">,
    Option<"elemType", "elem_t", "std::string", "\"i8\"", "The type of elem_t.">,
    Option<"accType", "acc_t", "std::string", "\"i32\"", "The type of acc_t.">
  ];
}
