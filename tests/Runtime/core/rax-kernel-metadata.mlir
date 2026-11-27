// RUN: split-file %s %t
// RUN: buddy-rax-kernel-metadata-test prepare %t
// RUN: rax-pack %t/metadata.mlir -o %t/plain.rax
// RUN: rax-inspect %t/plain.rax | FileCheck %s --check-prefix=INSPECT
// RUN: rax-pack %t/metadata.mlir -o %t/embedded.rax --embed-payload
// RUN: rax-inspect %t/embedded.rax | FileCheck %s --check-prefix=EMBED
// RUN: rm %t/optional.so
// RUN: buddy-rax-kernel-metadata-test plain %t/plain.rax | FileCheck %s --check-prefix=PLAIN
// RUN: env BUDDY_RAX_PAYLOAD_DIR=%t/cache buddy-rax-kernel-metadata-test embedded %t/embedded.rax | FileCheck %s --check-prefix=LOADED
// RUN: rax-pack %t/legacy.mlir -o %t/legacy.rax
// RUN: buddy-rax-kernel-metadata-test legacy %t/legacy.rax | FileCheck %s --check-prefix=LEGACY
// RUN: buddy-rax-kernel-metadata-test invalid %t | FileCheck %s --check-prefix=INVALID
// RUN: not rax-pack %t/bad-optional.mlir -o %t/bad.rax 2>&1 | FileCheck %s --check-prefix=BAD-OPTIONAL
// RUN: not rax-pack %t/bad-type.mlir -o %t/bad.rax 2>&1 | FileCheck %s --check-prefix=BAD-TYPE
// RUN: not rax-pack %t/bad-strides.mlir -o %t/bad.rax 2>&1 | FileCheck %s --check-prefix=BAD-STRIDES
// RUN: not rax-pack %t/bad-layout.mlir -o %t/bad.rax 2>&1 | FileCheck %s --check-prefix=BAD-LAYOUT

// INSPECT: @input  dtype=F32  shape=[2,4]  layout=RowMajor  strides=[4,1]
// INSPECT: @legacy  dtype=F32  shape=[2,4]  layout=Any  strides=[]
// INSPECT: @triton_softmax  kind=HostSharedLib  uri=file:optional.so
// INSPECT-NEXT: backend=triton_cpu  entry_symbol=buddy_external_kernel_query_v1
// INSPECT: kernel_abi = buddy.external.v1
// INSPECT: optional = true
// INSPECT: target = test-host
// EMBED: @triton_softmax  kind=HostSharedLib  uri=payload:optional.so
// EMBED: entry_symbol=buddy_external_kernel_query_v1
// EMBED: payload_footer:
// PLAIN: PASS plain RAX metadata round trip; mandatory dependencies=1; path=fallback
// LOADED: PASS embedded RAX metadata round trip; mandatory dependencies=1; path=fallback
// LEGACY: PASS legacy RAX defaults
// INVALID: PASS malformed runtime metadata rejected
// BAD-OPTIONAL: kernel_attrs.optional must be 'true' or 'false'
// BAD-TYPE: kernel_attrs values must be strings
// BAD-STRIDES: strides must contain one element stride per dimension
// BAD-LAYOUT: unsupported layout 'unknown'

//--- metadata.mlir
module {
  rhal.module @metadata attributes {version = "0.1.0", model_name = "softmax_metadata"} {
    // Optional comes first to verify it cannot become the primary mandatory library.
    rhal.codeobj @triton_softmax {
      id = 2 : i32, kind = "host_shared_lib", backend = "triton_cpu",
      uri = "file:optional.so", entry_symbol = "buddy_external_kernel_query_v1",
      kernel_attrs = {
        kernel_abi = "buddy.external.v1", operation = "softmax", dtype = "f32",
        layout = "row_major", shape = "2,4", strides = "4,1", axis = "1",
        target = "test-host", required_isa = "test-feature", optional = "true"
      }
    }
    rhal.codeobj @primary {id = 1 : i32, kind = "host_shared_lib", backend = "cpu", uri = "file:primary.so"}
    rhal.codeobj @dependency {id = 3 : i32, kind = "host_shared_lib", backend = "cpu", uri = "file:dependency.so", kernel_attrs = {optional = "false"}}
    rhal.buffer @input {space = "host", type = tensor<2x4xf32>, layout = "row_major", strides = array<i64: 4, 1>}
    rhal.buffer @legacy {space = "host", type = tensor<2x4xf32>}
    rhal.func @softmax {inputs = ["input"], outputs = ["legacy"], dispatch = "triton_softmax", args = ["input", "legacy"], scalars = array<i64: -1>}
  }
}

//--- legacy.mlir
module {
  rhal.module @legacy {
    rhal.codeobj @primary {id = 1 : i32, kind = "host_shared_lib", backend = "cpu", uri = "file:primary.so"}
    rhal.codeobj @dependency {id = 3 : i32, kind = "host_shared_lib", backend = "cpu", uri = "file:dependency.so"}
    rhal.buffer @input {space = "host", type = tensor<2x4xf32>}
    rhal.buffer @legacy {space = "host", type = tensor<2x4xf32>}
  }
}

//--- bad-optional.mlir
module {
  rhal.module @bad {
    rhal.codeobj @bad {id = 1 : i32, kind = "host_shared_lib", backend = "cpu", uri = "file:missing.so", kernel_attrs = {optional = "yes"}}
  }
}

//--- bad-type.mlir
module {
  rhal.module @bad {
    rhal.codeobj @bad {id = 1 : i32, kind = "host_shared_lib", backend = "cpu", uri = "file:missing.so", kernel_attrs = {axis = 1 : i64}}
  }
}

//--- bad-strides.mlir
module {
  rhal.module @bad {
    rhal.buffer @bad {space = "host", type = tensor<2x4xf32>, strides = array<i64: 4>}
  }
}

//--- bad-layout.mlir
module {
  rhal.module @bad {
    rhal.buffer @bad {space = "host", type = tensor<2x4xf32>, layout = "unknown"}
  }
}
