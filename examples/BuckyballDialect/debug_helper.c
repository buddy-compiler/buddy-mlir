#include <stdio.h>

// Shared by BuckyballDialect *-linux examples: call from MLIR with extui(cmpi)
// as i32. fail == 0 => pass, nonzero => fail.
void bb_test_report(int fail) { puts(fail ? "TEST FAILED" : "TEST PASS"); }
