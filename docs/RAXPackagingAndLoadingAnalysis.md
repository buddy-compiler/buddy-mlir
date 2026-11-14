# RAX Packaging and Runtime Loading

This document is the reference for the RAX file produced by `rax-pack` and
consumed by Buddy runtime tools. It covers the FlatBuffer manifest, the
optional embedded payload container, URI rewriting and extraction, and the
loading paths used by `buddy-cli` and `buddy-server`.

The source of truth remains the schema and implementations listed in
[Implementation references](#implementation-references). This document
describes their current behavior; it does not introduce a separate file-format
contract.

## End-to-end overview

```text
RHAL MLIR manifest
        |
        | rax-pack [-o model.rax] [--embed-payload]
        v
+------------------------------------------------------------+
| RAX0 FlatBuffer manifest                                   |
| + optional payload bytes + payload index + 64-byte footer  |
+------------------------------------------------------------+
        |
        | ModelManifest::loadFromRax()
        v
validated manifest + resolved resource paths
        |                                  |
        | buddy-cli                        | buddy-server
        v                                  v
runner_library plugin                 model_library plugin
InferenceRunner ABI                   descriptor + service ABI
```

The FlatBuffer manifest always describes the model. With
`--embed-payload`, selected files are appended to the RAX and their `file:`
URIs are rewritten to `payload:` URIs. At load time, payload resources are
extracted into a per-RAX cache before dynamic libraries or other consumers use
them.

## RHAL input

`rax-pack` accepts an MLIR file containing an `rhal.module`. A minimal example
with both runtime entry points is:

```mlir
module {
  rhal.module @example attributes {
      version = "0.1.0",
      model_name = "example",
      vocab_uri = "file:vocab.txt",
      runner_library = "file:example_runner.so",
      model_library = "file:example_server.so"} {
    rhal.constant @params {
      id = 1 : i32,
      storage = "external",
      type = tensor<1024xf32>,
      uri = "file:weights.bin"
    }
    rhal.codeobj @kernels {
      id = 1 : i32,
      kind = "host_shared_lib",
      backend = "cpu",
      uri = "file:example_model.so"
    }
  }
}
```

The two library attributes serve different hosts:

| Attribute | Consumer | Purpose |
| --- | --- | --- |
| `runner_library` | `buddy-cli` | Provides the `InferenceRunner` C ABI. |
| `model_library` | `buddy-server` | Provides the unified Server Model ABI descriptor and operation table. |

Both are module attributes stored as generic key-value entries. Adding
`model_library` did not require a change to `rax.fbs`.

`rax-pack` rejects additional module attributes whose names end in
`_library`. A server adapter must use the single `model_library` attribute;
model-specific server library attributes are not supported.

## FlatBuffer manifest

The base RAX file is a serialized `rhal.rax.Module` with the FlatBuffer file
identifier `RAX0`. The module contains:

| Field | Contents |
| --- | --- |
| `attrs` | String key-value attributes, including `model_name`, `vocab_uri`, `runner_library`, and `model_library`. |
| `requires` | Optional capability requirements. |
| `buffers` | Buffer bindings and tensor metadata. |
| `constants` | Inline or externally referenced constants. |
| `code_objects` | Host or device executable resources. |
| `functions` | Dispatch programs and their bindings. |

All files, including those with an appended payload, begin with this
FlatBuffer. The payload is not represented by a table in `rax.fbs`.

## Packaging modes

### Manifest-only RAX

```bash
build/bin/rax-pack input.mlir -o model.rax
```

Without `--embed-payload`, the output is only the `RAX0` FlatBuffer. Resource
URIs are preserved as written. In particular, `rax-pack` does not verify that
files referenced by `file:` URIs exist in this mode.

At runtime, relative `file:` paths are resolved from the directory containing
the RAX file. Keep the referenced files beside the RAX, or preserve the
relative directory layout encoded in the manifest.

### Self-contained RAX with payload

```bash
build/bin/rax-pack input.mlir -o model.rax --embed-payload
```

Payload mode embeds files referenced through `file:` URIs in these locations:

- every constant;
- every code object;
- the `vocab_uri` module attribute;
- the `runner_library` module attribute;
- the `model_library` module attribute.

Other module attributes are not embedded automatically, even when their value
looks like a URI. Existing `payload:` URIs and URIs with any other scheme are
left unchanged.

Relative source paths are resolved from the parent directory of the output
RAX, not from the input MLIR directory. Each selected source must exist and be
a regular file. Packaging fails otherwise. For example:

```text
/tmp/package/model.rax
/tmp/package/model.so
```

With output `/tmp/package/model.rax`, `file:model.so` reads
`/tmp/package/model.so` regardless of where `input.mlir` resides.

If `--embed-payload` finds no eligible `file:` URI, `rax-pack` emits a plain
FlatBuffer with no index or footer.

## URI rewriting and deduplication

For each embedded source, `rax-pack` replaces the manifest URI:

```text
file:example_model.so  ->  payload:example_model.so
```

Before deduplication, the source path is normalized with
`std::filesystem::weakly_canonical`. If canonicalization reports an error, the
absolute path is used instead. References with the same normalized source path
share one payload entry. This also means that one file used as both
`runner_library` and `model_library` is stored once.

Entry names are generated as follows:

1. Start with the source file's basename.
2. Replace every character outside `[A-Za-z0-9._-]` with `_`.
3. If the resulting name is empty, use `payload.bin`.
4. Resolve collisions by adding `_1`, `_2`, and so on.

Deduplication is path-based, not content-based. Two different paths with
identical bytes remain separate entries. When references of different kinds
deduplicate to one entry, the first reference determines the recorded payload
kind. The current runtime does not use that kind to select loading behavior.

## Appended payload layout

An embedded RAX is laid out as:

```text
+----------------------------+ offset 0
| RAX0 FlatBuffer manifest   |
+----------------------------+ manifest_size / payload_offset
| payload entry bytes        |
| ...                        |
+----------------------------+ index_offset
| payload index              |
+----------------------------+ file_size - 64
| payload footer (64 bytes)  |
+----------------------------+ file_size
```

Integers in the appended structures are little-endian. The writer adds no
alignment padding between regions or entries.

### Payload index

The index begins with:

```text
magic[8]       = "RAXIDX01"
version:u32    = 1
entry_count:u32
```

It is followed by `entry_count` variable-length records:

```text
kind:u16
reserved:u16
name_size:u32
file_offset:u64
file_size:u64
name[name_size]
```

`file_offset` is an absolute offset from the start of the RAX file. Payload
kinds currently have these numeric values:

| Value | Kind | Current use |
| ---: | --- | --- |
| 0 | `Unknown` | Unclassified resource. |
| 1 | `Constant` | Constant data. |
| 2 | `CodeObject` | Code object and `runner_library` files. |
| 3 | `Vocab` | `vocab_uri` files. |
| 4 | `ModelPlugin` | `model_library` files. |

### Payload footer

The last 64 bytes contain:

```text
magic[8]          = "RAXPAY01"
manifest_size:u64
payload_offset:u64
payload_size:u64
index_offset:u64
index_size:u64
flags:u64         # currently 1: payload present
reserved:u64      # currently 0
```

The footer lets a reader locate the manifest and index without scanning the
payload. A file whose last 64 bytes do not start with `RAXPAY01` is treated as
a manifest-only RAX.

## Manifest validation and URI resolution

`ModelManifest::loadFromRax` performs these steps:

1. Convert the RAX path to an absolute path and read its size.
2. Inspect the final 64 bytes for the `RAXPAY01` footer.
3. If the footer exists, validate the manifest, payload, and index ranges and
   require `payload_offset == manifest_size`.
4. Verify the `RAX0` identifier and the FlatBuffer contents.
5. If present, parse the `RAXIDX01` index and reject unsupported versions,
   duplicate names, truncated records, and out-of-range entries.
6. Resolve code objects, constants, and module attributes.

The manifest must contain at least one code object. Duplicate
`model_library` module attributes are rejected explicitly.

URI resolution follows these rules:

| URI | Result |
| --- | --- |
| `file:X` | `X` resolved relative to the parent directory of the RAX file. |
| `payload:X` | Entry `X` extracted from the appended payload and returned as an absolute cache path. |
| Any other scheme | Original string passed through unchanged. |

Payload entry names used by a URI must be non-empty and must not contain
`..`, `/`, or `\`. A `payload:` URI is an error when the footer/index is
absent or when the named entry is missing.

The returned `ModelManifest` exposes both generic and compatibility-oriented
views of the resolved data:

| Output | Source |
| --- | --- |
| `codeObjects` | All code objects with original URI and resolved path. |
| `soPath` | First `HostSharedLib` code object. |
| `dependentSoPaths` | Remaining `HostSharedLib` code objects. |
| `constants` | All constants with original URI and resolved path. |
| `weightPaths` | Resolved paths for external constants. |
| `vocabPath` | Resolved `vocab_uri`. |
| `runnerLibraryPath` | Resolved `runner_library`. |
| `modelLibraryPath` | Resolved `model_library`. |
| `moduleAttrs` | Original string values for all module attributes. |
| `resolvedModuleAttrs` | Resolved values for URI-like module attributes. |

## Payload extraction cache

The extraction cache root is selected in this order:

1. Non-empty `$BUDDY_RAX_PAYLOAD_DIR`.
2. `$HOME/.buddy/rax_payload` when `$HOME` is set.
3. `buddy_rax_payload` under the system temporary directory.

Each RAX gets a subdirectory derived from
`std::hash(absolute-path:file-size:mtime)`. A resource is extracted to its
payload entry name within that directory.

When the destination is already a regular file of the expected size, it is
reused. Otherwise, the runtime writes `<entry>.tmp`, closes it, and renames it
to the final path. Cache reuse currently compares only the file size; the
payload format and loader do not currently store or validate a checksum for
embedded entries.

The extraction directory must reside on a filesystem that permits the
intended operation. In particular, dynamically loaded `.so` files require a
location from which the platform loader can map executable code.

## `buddy-cli` loading flow

The recommended invocation is:

```bash
build/bin/buddy-cli --model model.rax --prompt "Hello"
```

The relevant loading sequence is:

```text
buddy-cli --model model.rax
  -> ModelManifest::loadFromRax(model.rax)
  -> select runnerLibraryPath
  -> dlopen(path, RTLD_NOW | RTLD_LOCAL)
  -> resolve buddy_create_inference_runner_v1
  -> resolve buddy_destroy_inference_runner_v1
  -> create InferenceRunner
  -> runner->run(RunConfig{raxPath = original model.rax, ...})
  -> destroy InferenceRunner and close the library
```

An explicit `--runner-so` overrides `runner_library` from the manifest. The
CLI also retains legacy model arguments and may select a built-in runner when
no runner plugin path is available, depending on build configuration and
model name. The RAX path passed to the runner remains the original file; the
runner can load the same manifest to find model code, constants, and vocab.

## `buddy-server` loading flow

The server accepts the model package through:

```bash
build/bin/buddy-server --model model.rax
```

Unlike `buddy-cli`, it does not accept a separate model plugin path. The
current loading sequence is:

```text
buddy-server --model model.rax
  -> ModelManifest::loadFromRax(model.rax)
  -> require modelLibraryPath
  -> load RAX dependency libraries globally
  -> ServerModelPluginHandle(path)
       -> dlopen(path, RTLD_NOW | RTLD_LOCAL)
       -> resolve buddy_server_model_descriptor_v1
       -> if absent, detect buddy_model_plugin_descriptor_v1 and report that
          the package must be regenerated
       -> validate the descriptor and unified operation table
  -> create one model instance through the unified operation table
  -> asynchronously load the model with resolved RAX resources
  -> register routes and dispatch the operations declared by operationMask
```

The descriptor keeps service classification separate from the operations,
input modalities, execution capabilities, and result representation. The
service kind remains one of:

- `Resident`
- `Embedding`
- `MaskedLM`
- `Transcription`

All service kinds use the same `create`, `load`, `getStatus`, `invoke`, and
`destroy` operation table. `buddy-server` no longer probes service-specific
factory symbols or reopens the plugin through typed handles.

## Inspection and troubleshooting

Inspect a package without running the model:

```bash
build/bin/rax-inspect model.rax
```

`rax-inspect` prints module attributes, constants, code objects, functions,
and, when present, the footer and every payload index record with its kind,
name, offset, and size. It validates the manifest and the basic appended
container structure. It does not extract payload entries or validate a model
plugin's dynamic ABI.

Common failures:

| Symptom | Check |
| --- | --- |
| `missing payload source` from `rax-pack` | Resolve the `file:` URI from the output RAX directory and ensure it is a regular file. |
| `payload URI found but payload section missing` | Repack with `--embed-payload`, or change the URI back to a deployable `file:` path. |
| `payload entry not found` | Compare the rewritten manifest URI with the index shown by `rax-inspect`. |
| `dlopen` failure | Check the extracted/file URI path, dependent shared libraries, permissions, and executable-mapping policy. |
| Server reports missing `model_library` | Add the server plugin URI to the `rhal.module` attributes and repack. |
| Server reports the legacy server ABI | Regenerate or download a current RAX package containing `buddy_server_model_descriptor_v1`. |
| Descriptor ABI error | Rebuild the server plugin and `buddy-server` from compatible runtime headers. |

## Compatibility and limitations

- Manifest-only RAX files remain supported. Payload support is detected from
  the footer rather than from a schema version.
- Appended payload data is outside the FlatBuffer schema. Consumers that read
  the entire file as one FlatBuffer should instead limit verification to
  `manifest_size` when a payload footer is present.
- The payload index records a kind but the current runtime resolves entries by
  name, independent of kind.
- Payload entries have no checksum. Both packaging deduplication and runtime
  cache identity are metadata/path based rather than content-addressed.
- Only the explicitly listed URI fields are embedded. New resource-bearing
  attributes require corresponding packer and loader support.
- `file:` URI handling is local-path handling; URI fragments and percent
  decoding are not interpreted specially.

## Implementation references

| Area | Source |
| --- | --- |
| FlatBuffer schema | `runtime/rax/rax.fbs` |
| RHAL-to-RAX packaging and payload writer | `tools/rax-pack/rax-pack.cpp` |
| Manifest validation, URI resolution, and extraction | `runtime/include/buddy/runtime/core/ModelManifest.h` |
| CLI runner loading | `tools/buddy-cli/buddy-cli.cpp` |
| Server entry point | `tools/buddy-server/src/buddy-server.cpp` |
| Server plugin loading | `tools/buddy-server/src/plugin/ServerModelPluginHandle.cpp` |
| Package inspection | `tools/rax-inspect/rax-inspect.cpp` |
