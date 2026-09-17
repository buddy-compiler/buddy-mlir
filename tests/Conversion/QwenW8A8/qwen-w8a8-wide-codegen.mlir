// RUN: %PYTHON %S/Inputs/check-fpga-wide.py buddy-opt buddy-translate llc %t
// The helper checks the full emitted schedules, including all output tile
// offsets and all accumulator lanes, then compiles those same programs at
// O0/O2/O3 and inspects both post-allocation MIR and final assembly.
module {}
