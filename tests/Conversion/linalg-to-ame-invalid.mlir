// RUN: not buddy-opt %s --lower-linalg-to-ame='target=unknown' 2>&1 | FileCheck %s

module {}

// CHECK: unsupported AME target: unknown; expected generic or gem5
