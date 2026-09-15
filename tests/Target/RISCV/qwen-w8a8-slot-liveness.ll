; RUN: %python %S/Inputs/check-fpga-slots.py buddy-llc
; Exercise CFG joins, destructive updates, dead accelerator writes and calls.
; Matrix slots must be validated before eliminating PHIs or running ordinary RA.
