#!/usr/bin/env python3
# ===- boscame_sim.py ------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------

"""Interpret emitted BOSCAME modules on a software NH/E6 datapath model (DDR args, two SPM banks, DMA contract).

Inputs: MLIR module path, optional --function/--buffers/--state.
Outputs: Printed/returned buffer state and recorded DDR touch sets; fails closed on unknown ops.
When used: Lit and example golden checks before or instead of board runs.

The pass emits a small, regular subset of MLIR: `memref`, `arith`, `scf`,
`linalg` and the `bosc_ame` tile/DMA/SPM operations.  This module interprets
exactly that subset on a software model of the NH/e6 datapath:

* DDR is modelled as one byte array per function argument, so a test can seed
  inputs and read results back bit for bit.
* SPM is modelled as the two banks the hardware has, with the 128 KiB capacity
  checked on every access (a bank overrun is reported, not silently ignored).
* `bosc_ame.dma.start` performs the transfer only when the hardware contract
  holds: one contiguous byte range, both endpoints 64 B aligned, `bytes` a
  positive multiple of 64, and no request outstanding on the channel.  Every
  DDR byte the transfer reads or writes is recorded.
* Every DDR access (scalar or DMA) is recorded, so a test can assert that the
  addresses a mask excludes are never touched.

It is deliberately a *test* tool: it models semantics, not performance, and it
refuses anything it does not understand instead of guessing.
"""

import argparse
import json
import re
import sys

SPM_BANK_BYTES = 128 * 1024
ELEMENT_BYTES = {
    "i8": 1,
    "i16": 2,
    "i32": 4,
    "i64": 8,
    "index": 8,
    "f32": 4,
    "f64": 8,
}


class SimError(Exception):
    """Anything the simulator cannot execute falls out here."""


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------


class ScalarType:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


class VectorType:
    def __init__(self, shape, element):
        self.shape = shape
        self.element = element

    def __repr__(self):
        return "vector<{}x{}>".format(
            "x".join(str(s) for s in self.shape), self.element
        )


class MemRefType:
    def __init__(self, shape, element, offset, strides, spm_bank, dynamic):
        self.shape = shape
        self.element = element
        self.offset = offset
        # `strides` and `shape` hold None for dynamic entries.
        self.strides = strides
        self.spm_bank = spm_bank
        self.dynamic = dynamic
        self.element_bytes = ELEMENT_BYTES.get(element)
        if self.element_bytes is None:
            raise SimError("unknown element type " + element)

    def __repr__(self):
        return "memref<{}x{}>".format(
            "x".join(str(s) for s in self.shape), self.element
        )


class TokenType:
    def __init__(self, channel):
        self.channel = channel


def _split_top_level(text, separator=","):
    """Split on `separator`, ignoring separators inside nested syntax."""
    parts = []
    depth = 0
    current = ""
    for index, char in enumerate(text):
        if char in "<([{":
            depth += 1
        elif char in ">)]}" and not (
            char == ">" and index > 0 and text[index - 1] == "-"
        ):
            depth -= 1
        if char == separator and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    if current:
        parts.append(current)
    return parts


def parse_shape(text):
    """`64x16` / `?x16` -> ([64, 16], False)."""
    dynamic = False
    shape = []
    for piece in text.split("x"):
        if piece == "?":
            dynamic = True
            shape.append(None)
        else:
            shape.append(int(piece))
    return shape, dynamic


def _strip_layout(layout):
    """Return the `#bosc_ame.spm<N>` bank of a memref layout, when present."""
    match = re.search(r"#bosc_ame\.spm<(\d+)>", layout or "")
    return int(match.group(1)) if match else None


def parse_type(text):
    """Parse the subset of MLIR types the pass emits."""
    text = text.strip()
    if text.startswith("memref<"):
        inner = text[len("memref<") : -1]
        head, separator, layout = inner.partition(",")
        layout = (separator + layout) if separator else ""
        body = head.strip()
        offset = 0
        strides = None
        if body.startswith("strided<"):
            fields = _split_top_level(body[len("strided<") : -1])
            body = fields[0].strip()
            if len(fields) > 1:
                strides = [
                    None if piece == "?" else int(piece)
                    for piece in fields[1].strip().split("x")
                ]
            for field in fields[2:]:
                if field.strip().startswith("offset:"):
                    offset = int(field.split(":")[1])
        pieces = body.split("x")
        element = pieces[-1]
        shape, dynamic = parse_shape("x".join(pieces[:-1]))
        return MemRefType(
            shape, element, offset, strides, _strip_layout(layout), dynamic
        )
    if text.startswith("vector<"):
        inner = text[len("vector<") : -1]
        if inner.startswith("["):
            raise SimError("scalable vectors are not executable here: " + text)
        pieces = inner.split("x")
        shape, _ = parse_shape("x".join(pieces[:-1]))
        return VectorType(shape, pieces[-1])
    if text.startswith("!bosc_ame.dma_token<"):
        return TokenType(int(text[len("!bosc_ame.dma_token<") : -1]))
    if text.startswith("!tt."):
        raise SimError("Triton types are not executable here: " + text)
    return ScalarType(text)


def parse_result_type(op_types):
    """`vector<A>, vector<B> -> vector<C>` -> `vector<C>`."""
    text = op_types.split("->")[-1].strip()
    if not text:
        raise SimError("operation without a result type")
    return parse_type(text)


def types_before_arrow(op_types):
    text = op_types.split("->")[0]
    return [
        parse_type(piece) for piece in _split_top_level(text) if piece.strip()
    ]


# --------------------------------------------------------------------------
# Values
# --------------------------------------------------------------------------


class MemRef:
    """A descriptor: a byte buffer plus offset, sizes and element strides."""

    def __init__(
        self, data, offset, sizes, strides, element, spm_bank, name=None
    ):
        self.data = data
        self.offset = offset
        self.sizes = sizes
        self.strides = strides
        self.element = element
        self.spm_bank = spm_bank
        self.name = name
        self.element_bytes = ELEMENT_BYTES.get(element)
        if self.element_bytes is None:
            raise SimError("unknown element type " + element)

    def clone(self, offset=None, sizes=None, strides=None, element=None):
        return MemRef(
            self.data,
            self.offset if offset is None else offset,
            self.sizes if sizes is None else sizes,
            self.strides if strides is None else strides,
            self.element if element is None else element,
            self.spm_bank,
            self.name,
        )

    def byte_address(self, indices):
        if len(indices) != len(self.sizes):
            raise SimError(f"rank mismatch on {self}")
        address = self.offset
        for index, stride in zip(indices, self.strides):
            address += index * stride * self.element_bytes
        return address

    def store(self, indices, value, machine):
        address = self.byte_address(indices)
        self.store_byte_address(address, value, machine)

    def store_byte_address(self, address, value, machine, direct=False):
        """Store an element at a byte address supplied by an AME operation.

        Descriptor-based CPU stores use ``store(indices, ...)``.  The AME
        ``msce32.m`` instruction instead advances each row by its explicit
        byte stride, so it needs this small escape hatch to model flat pointer
        views without changing the descriptor's logical strides.
        """
        self._check_bounds(address, machine)
        if direct:
            machine.record_direct_access(self, address, self.element_bytes)
        else:
            machine.record_access(self, address, self.element_bytes, write=True)
        for i in range(self.element_bytes):
            self.data[address + i] = (
                (value >> (8 * i)) & 0xFF
                if value >= 0
                else ((value + (1 << (8 * self.element_bytes))) >> (8 * i))
                & 0xFF
            )

    def load(self, indices, machine):
        address = self.byte_address(indices)
        self._check_bounds(address, machine)
        machine.record_access(self, address, self.element_bytes, write=False)
        value = 0
        for i in range(self.element_bytes):
            value |= self.data[address + i] << (8 * i)
        if self.element.startswith("i") and self.element != "index":
            bits = self.element_bytes * 8
            if value >= (1 << (bits - 1)):
                value -= 1 << bits
        return value

    def _check_bounds(self, address, machine):
        if address < 0 or address + self.element_bytes > len(self.data):
            raise SimError(
                f"out of bounds access at byte {address} of a {len(self.data)}-byte "
                "buffer"
            )
        if self.spm_bank is not None and len(self.data) > SPM_BANK_BYTES:
            raise SimError(
                f"SPM bank {self.spm_bank} is larger than the hardware bank"
            )

    def byte_span(self, machine):
        """(first, past-last) absolute byte of the descriptor's elements."""
        first = self.byte_address([0] * len(self.sizes))
        last = self.byte_address([s - 1 for s in self.sizes])
        return first, last + self.element_bytes


class DmaToken:
    def __init__(self, channel, source, destination, length, machine):
        self.channel = channel
        self.source = source
        self.destination = destination
        self.length = length
        self.machine = machine


# --------------------------------------------------------------------------
# Parsing the module into a tree of operations
# --------------------------------------------------------------------------


class Op:
    def __init__(self, text, line):
        self.text = text
        self.line = line
        self.blocks = []
        self.result_type = None

    def location(self):
        return f"line {self.line}"


def read_module(path):
    with open(path) as handle:
        lines = handle.readlines()
    ops, index = read_block(lines, 0, len(lines), top=True)
    if index != len(lines):
        raise SimError("unbalanced braces")
    return ops


def read_block(lines, index, end, top=False):
    """Read operations until the closing brace, which is left unconsumed."""
    ops = []
    while index < end:
        raw = lines[index]
        text = raw.split("//")[0].strip()
        index += 1
        if not text:
            continue
        if text == "}" or text.startswith("} {") or text == "} else {":
            return ops, index - 1
        if text.endswith("{"):
            head = text[:-1].strip()
            children, index = read_block(lines, index, end)
            op = Op(head, index + 1)
            op.blocks = [children]
            if (
                index < end
                and lines[index].split("//")[0].strip() == "} else {"
            ):
                index += 1
                else_children, index = read_block(lines, index, end)
                op.blocks.append(else_children)
            if index < end and (
                lines[index].split("//")[0].strip() == "}"
                or lines[index].split("//")[0].strip().startswith("} {")
            ):
                index += 1
            else:
                raise SimError(
                    "expected '}' after block starting with " + repr(head[:70])
                )
            if head.startswith("module"):
                ops.extend(children)
            else:
                ops.append(op)
            continue
        ops.append(Op(text, index))
    return ops, index


# --------------------------------------------------------------------------
# The machine
# --------------------------------------------------------------------------


class Machine:
    def __init__(self):
        self.ddr = {}
        self.spm = [bytearray(SPM_BANK_BYTES), bytearray(SPM_BANK_BYTES)]
        self.values = {}
        self.outstanding = {}
        self.accesses = []
        self.dma_log = []
        self.ddr_policy = None

    # -- DDR access recording -------------------------------------------
    def record_access(self, memref, address, size, write):
        if memref.spm_bank is not None:
            return
        required = "write" if write else "read"
        if self.ddr_policy != required:
            raise SimError(
                f"CPU DDR {required} requires the {required} policy; "
                f"active policy is {self.ddr_policy!r}"
            )
        self.accesses.append(
            {
                "buffer": memref.name,
                "start": address,
                "end": address + size,
                "kind": "write" if write else "read",
            }
        )

    def record_direct_access(self, memref, address, size):
        """Record a hardware AME DDR write without a CPU policy call."""
        if memref.spm_bank is not None:
            raise SimError("AME direct store cannot target an SPM window")
        self.accesses.append(
            {
                "buffer": memref.name,
                "start": address,
                "end": address + size,
                "kind": "write",
            }
        )

    def record_dma(
        self, source, destination, source_address, destination_address, length
    ):
        for memref, address in (
            (source, source_address),
            (destination, destination_address),
        ):
            if memref.spm_bank is None:
                self.accesses.append(
                    {
                        "buffer": memref.name,
                        "start": address,
                        "end": address + length,
                        "kind": "write" if memref is destination else "read",
                    }
                )
        self.dma_log.append(
            {
                "channel": None,
                "source": {
                    "buffer": source.name,
                    "start": source_address,
                    "end": source_address + length,
                },
                "destination": {
                    "buffer": destination.name,
                    "start": destination_address,
                    "end": destination_address + length,
                },
                "length": length,
            }
        )

    def reads(self):
        return [a for a in self.accesses if a["kind"] == "read"]

    def writes(self):
        return [a for a in self.accesses if a["kind"] == "write"]


# --------------------------------------------------------------------------
# Interpretation
# --------------------------------------------------------------------------


def strip_prefix(text, prefix):
    if not text.startswith(prefix):
        raise SimError(f"expected '{prefix}' in {text!r}")
    return text[len(prefix) :].strip()


def split_result(text):
    """`%r = arith.addi ...` -> ("%r", "arith.addi ...").

    Only a leading result name (optionally `%r:2`) counts: operations such as
    `scf.for %i = ...` carry `=` inside their operands.
    """
    match = re.match(r"^((?:%[A-Za-z0-9_.\-]+)(?::[0-9]+)?)\s*=\s*(.*)$", text)
    if match:
        return match.group(1), match.group(2)
    return None, text


def op_name(text):
    return split_result(text)[1].split(" ")[0].split("(")[0]


def result_name(text):
    return split_result(text)[0]


def after_name(text):
    return split_result(text)[1]


def parse_operands(body, count):
    """Collect `%a, %b, ...` operands; tolerates attributes and `ins(...)`."""
    body = re.sub(r"\{[^}]*\}", "", body, count=1)
    names = []
    for token in re.findall(r"%[A-Za-z0-9_.\-]+", body):
        if token not in names:
            names.append(token)
    return names


def split_op_and_types(text):
    """Split an operation line into its body and its trailing type list."""
    body = text
    depth = 0
    for index, char in enumerate(text):
        if char in "<([{":
            depth += 1
        elif char in ">)]}" and not (
            char == ">" and index > 0 and text[index - 1] == "-"
        ):
            depth -= 1
        elif char == ":" and depth == 0:
            return text[:index].strip(), text[index + 1 :].strip()
    return body, ""


def value_of(machine, name):
    name = name.strip()
    if name not in machine.values:
        raise SimError("use of an undefined SSA value " + name)
    return machine.values[name]


def constant_of(machine, name):
    value = value_of(machine, name.strip())
    if not isinstance(value, (int, float)):
        raise SimError("expected a scalar in " + name)
    return value


def run_op(op, machine, args):
    """Execute one operation, reporting the source line on any failure."""
    try:
        return _run_op_impl(op, machine, args)
    except SimError:
        raise
    except _ReturnSignal:
        raise
    except Exception as error:  # noqa: BLE001 - reported, not swallowed
        raise SimError(f"{error} while executing {op.text!r}") from error


def _run_op_impl(op, machine, args):
    text = op.text
    name = op_name(text)
    body = after_name(text)
    result = result_name(text)
    op_body, op_types = split_op_and_types(body)

    # ---- arith ------------------------------------------------------
    if name == "arith.constant":
        payload = op_body[len("arith.constant") :].strip()
        if payload.startswith("dense<") or payload.startswith("splat<"):
            raise SimError("only scalar constants are executable: " + text)
        machine.values[result] = int(payload)
        return
    if name.startswith("arith."):
        operands = parse_operands(op_body, 0)
        try:
            scalars = [constant_of(machine, operand) for operand in operands]
        except SimError as error:
            raise SimError(f"{error} in {text!r}") from error
        if name == "arith.addi":
            machine.values[result] = scalars[0] + scalars[1]
        elif name == "arith.subi":
            machine.values[result] = scalars[0] - scalars[1]
        elif name == "arith.muli":
            machine.values[result] = scalars[0] * scalars[1]
        elif name == "arith.divui":
            machine.values[result] = scalars[0] // scalars[1]
        elif name == "arith.divsi":
            machine.values[result] = int(scalars[0] / scalars[1])
        elif name == "arith.remui":
            machine.values[result] = scalars[0] % scalars[1]
        elif name == "arith.andi":
            machine.values[result] = scalars[0] & scalars[1]
        elif name == "arith.ori":
            machine.values[result] = scalars[0] | scalars[1]
        elif name == "arith.xori":
            machine.values[result] = scalars[0] ^ scalars[1]
        elif name == "arith.shli":
            machine.values[result] = scalars[0] << scalars[1]
        elif name == "arith.shrsi":
            machine.values[result] = scalars[0] >> scalars[1]
        elif (
            name == "arith.index_cast"
            or name == "arith.extsi"
            or name == "arith.extui"
            or name == "arith.trunci"
        ):
            machine.values[result] = scalars[0]
        elif name == "arith.cmpi":
            predicate = op_body.split(" ")[1].rstrip(",")
            lhs = scalars[0]
            rhs = scalars[1]
            machine.values[result] = int(
                {
                    "eq": lhs == rhs,
                    "ne": lhs != rhs,
                    "slt": lhs < rhs,
                    "sle": lhs <= rhs,
                    "sgt": lhs > rhs,
                    "sge": lhs >= rhs,
                    "ult": lhs < rhs,
                    "ule": lhs <= rhs,
                    "ugt": lhs > rhs,
                    "uge": lhs >= rhs,
                }[predicate]
            )
        elif name == "arith.select":
            machine.values[result] = scalars[1] if scalars[0] else scalars[2]
        elif name == "arith.maxsi":
            machine.values[result] = max(scalars)
        elif name == "arith.minsi":
            machine.values[result] = min(scalars)
        else:
            raise SimError("unsupported arith op: " + text)
        return

    # ---- memref -----------------------------------------------------
    if name == "memref.view":
        match = re.match(
            r"memref\.view (%\S+)\[(%\S+)\]\[\]\s*:\s*(.*?)\s+to\s+(.*)$", body
        )
        if not match:
            raise SimError("unsupported memref.view: " + text)
        source = value_of(machine, match.group(1))
        offset = constant_of(machine, match.group(2))
        view_type = parse_type(match.group(4))
        # The offset of `memref.view` is in bytes of the source element type.
        absolute = source.offset + offset * source.element_bytes
        sizes = [
            s if s is not None else len(source.data) // view_type.element_bytes
            for s in view_type.shape
        ]
        strides = view_type.strides or _contiguous_strides(sizes)
        machine.values[result] = MemRef(
            source.data,
            absolute,
            sizes,
            strides,
            view_type.element,
            view_type.spm_bank,
            source.name,
        )
        return
    if name == "memref.reinterpret_cast":
        match = re.match(
            r"memref\.reinterpret_cast (%\S+) to offset: \[(.*?)\], "
            r"sizes: \[(.*?)\], strides: \[(.*?)\]\s*:\s*(.*?) to (.*)$",
            body,
        )
        if not match:
            raise SimError("unsupported reinterpret_cast: " + text)
        source = value_of(machine, match.group(1))
        offset = int(match.group(2))
        sizes = [int(part) for part in _split_top_level(match.group(3))]
        strides = [int(part) for part in _split_top_level(match.group(4))]
        target_type = parse_type(match.group(6))
        machine.values[result] = MemRef(
            source.data,
            source.offset + offset * target_type.element_bytes,
            sizes,
            strides,
            target_type.element,
            target_type.spm_bank,
            source.name,
        )
        return
    if name == "memref.subview":
        match = re.match(
            r"memref\.subview (%\S+)\[(.*?)\] \[(.*?)\] \[(.*?)\]\s*:\s*"
            r"(.*?) to (.*)$",
            body,
        )
        if not match:
            raise SimError("unsupported subview: " + text)
        source = value_of(machine, match.group(1))
        offsets = [int(part) for part in _split_top_level(match.group(2))]
        sizes = [int(part) for part in _split_top_level(match.group(3))]
        steps = [int(part) for part in _split_top_level(match.group(4))]
        target_type = parse_type(match.group(6))
        absolute = source.offset
        for index, offset in enumerate(offsets):
            absolute += offset * source.strides[index] * source.element_bytes
        strides = [step * stride for step, stride in zip(steps, source.strides)]
        machine.values[result] = MemRef(
            source.data,
            absolute,
            sizes,
            strides,
            target_type.element,
            target_type.spm_bank,
            source.name,
        )
        return
    if name == "memref.load":
        # Hardware rule the board established: the E6 CPU cannot complete an
        # access to the SPM window, so a load from SPM is a bug in the
        # lowered code, not a slow path.
        if "bosc_ame.spm" in text:
            raise SimError(
                "the CPU cannot load from an SPM window; only DMA and the AME"
                " touch SPM: " + text
            )
        match = re.match(r"memref\.load (%\S+)\[(.*?)\]\s*:\s*(.*)$", body)
        if not match:
            raise SimError("unsupported load: " + text)
        memref = value_of(machine, match.group(1))
        indices = [
            constant_of(machine, part)
            for part in _split_top_level(match.group(2))
        ]
        machine.values[result] = memref.load(indices, machine)
        return
    if name == "memref.store":
        match = re.match(
            r"memref\.store (%\S+), (%\S+)\[(.*?)\]\s*:\s*(.*)$", body
        )
        if not match:
            raise SimError("unsupported store: " + text)
        value = constant_of(machine, match.group(1))
        memref = value_of(machine, match.group(2))
        if memref.spm_bank is not None:
            raise SimError("the CPU cannot store to an SPM window: " + text)
        indices = [
            constant_of(machine, part)
            for part in _split_top_level(match.group(3))
        ]
        memref.store(indices, value, machine)
        return
    if name.startswith("memref."):
        raise SimError("unsupported memref op: " + text)

    # ---- linalg -----------------------------------------------------
    if name == "linalg.fill":
        match = re.match(
            r"linalg\.fill ins\((%\S+) : .*?\) outs\((%\S+)", op_body
        )
        if not match:
            raise SimError("unsupported linalg.fill: " + text)
        scalar = constant_of(machine, match.group(1))
        target = value_of(machine, match.group(2))
        if target.spm_bank is not None:
            raise SimError("the CPU cannot fill an SPM window: " + text)
        for index in _iterate(target.sizes):
            target.store(index, scalar, machine)
        return
    if name == "linalg.generic":
        interpret_linalg_generic(op, machine, op_body)
        return
    if name == "linalg.yield":
        machine.yielded = [
            value_of(machine, token) for token in parse_operands(op_body, 0)
        ]
        return

    # ---- scf --------------------------------------------------------
    if name == "scf.for":
        interpret_scf_for(op, machine, op_body)
        return
    if name == "scf.if":
        interpret_scf_if(op, machine)
        return
    if name == "scf.yield":
        machine.yielded = [
            machine.values.get(token) for token in parse_operands(op_body, 0)
        ]
        return
    if name.startswith("scf."):
        raise SimError("unsupported scf op: " + text)

    # ---- bosc_ame ---------------------------------------------------
    if name == "bosc_ame.spm.view":
        attributes = re.search(r"\{([^}]*)\}", body)
        fields = dict(re.findall(r"(\w+) = ([0-9]+)", attributes.group(1)))
        bank = int(fields["bank"])
        offset = int(fields["offset"])
        target_type = parse_result_type(op_types)
        sizes = [s for s in target_type.shape]
        machine.values[result] = MemRef(
            machine.spm[bank],
            offset,
            sizes,
            target_type.strides or _contiguous_strides(sizes),
            target_type.element,
            bank,
            f"spm{bank}",
        )
        return
    if name in (
        "bosc_ame.msce32.m",
        "bosc_ame.intr.msce32.m",
        "ame.msce32.m",
    ):
        # Model the direct AME accumulator write before LLVM lowers it to the
        # target intrinsic.  The stride is expressed in bytes and therefore
        # does not necessarily match the destination memref's logical row
        # stride (a flat descriptor is valid too).
        operands = parse_operands(op_body, 0)
        if len(operands) < 3:
            raise SimError("msce32.m expects accumulator, destination, stride")
        matrix = value_of(machine, operands[0])
        destination = value_of(machine, operands[1])
        stride = constant_of(machine, operands[2])
        if machine.outstanding:
            raise SimError("direct msce32.m before DMA wait")
        if destination.spm_bank is not None:
            raise SimError("direct msce32.m destination must be external DDR")
        if (
            not isinstance(matrix, list)
            or not matrix
            or not isinstance(matrix[0], list)
            or not matrix[0]
            or any(
                not isinstance(row, list) or len(row) != len(matrix[0])
                for row in matrix
            )
        ):
            raise SimError("msce32.m requires a rectangular accumulator")
        rows, columns = len(matrix), len(matrix[0])
        stride = int(stride)
        row_bytes = columns * destination.element_bytes
        if stride < row_bytes or stride % destination.element_bytes:
            raise SimError("invalid msce32.m byte stride")
        if len(destination.sizes) == 2 and (
            rows > destination.sizes[0] or columns > destination.sizes[1]
        ):
            raise SimError("msce32.m tile exceeds destination shape")
        base = destination.byte_address([0] * len(destination.sizes))
        for row, values in enumerate(matrix):
            row_start = base + row * stride
            if row_start < 0 or row_start + row_bytes > len(destination.data):
                raise SimError("msce32.m writes outside backing storage")
            for column, value in enumerate(values):
                destination.store_byte_address(
                    row_start + column * destination.element_bytes,
                    value,
                    machine,
                    direct=True,
                )
        return
    if name.startswith("bosc_ame.tile."):
        interpret_tile_op(op, machine, op_body, op_types)
        return
    if name == "bosc_ame.dma.start":
        machine.ddr_policy = "write"
        match = re.match(
            r"bosc_ame\.dma\.start (%\S+), (%\S+), (%\S+)\s*\{([^}]*)\}",
            op_body,
        )
        if not match:
            raise SimError("unsupported dma.start: " + text)
        channel = int(re.search(r"channel = (\d+)", match.group(4)).group(1))
        if channel in machine.outstanding:
            raise SimError(
                f"channel {channel} already has an outstanding request; the "
                "hardware allows one"
            )
        source = value_of(machine, match.group(1))
        destination = value_of(machine, match.group(2))
        length = constant_of(machine, match.group(3))
        if length <= 0 or length % 64 != 0:
            raise SimError(
                f"DMA length {length} is not a positive multiple of 64"
            )
        source_span = source.byte_span(machine)
        destination_span = destination.byte_span(machine)
        if source_span[0] % 64 or destination_span[0] % 64:
            raise SimError(
                f"DMA endpoint is not 64 B aligned: {source_span} -> {destination_span}"
            )
        if length > source_span[1] - source_span[0]:
            raise SimError("DMA reads past its source")
        if length > destination_span[1] - destination_span[0]:
            raise SimError("DMA writes past its destination")
        machine.record_dma(
            source, destination, source_span[0], destination_span[0], length
        )
        machine.values[result] = DmaToken(
            channel, source, destination, length, machine
        )
        machine.outstanding[channel] = result

        return
    if name == "bosc_ame.dma.wait":
        token = value_of(machine, parse_operands(op_body, 0)[0])
        if not isinstance(token, DmaToken):
            raise SimError("dma.wait of a non-token")
        if machine.outstanding.get(token.channel) is None:
            raise SimError("dma.wait without an outstanding request")
        del machine.outstanding[token.channel]
        # The engine copies one contiguous byte range; it completes here, and
        # SPM/DMA ordering in the pass waits immediately, so modelling the copy
        # at `wait` matches the hardware contract.
        start = token.source.byte_span(machine)[0]
        target = token.destination.byte_span(machine)[0]
        token.destination.data[target : target + token.length] = bytes(
            token.source.data[start : start + token.length]
        )
        machine.ddr_policy = "read"
        return
    if name == "bosc_ame.dma.abi":
        raise SimError("the module still carries a dma.abi marker operation")
    if name.startswith("bosc_ame."):
        raise SimError("unsupported bosc_ame op: " + text)

    if name in ("func.call", "call"):
        match = re.fullmatch(
            r"(?:func\.)?call @bosc_ame_ddr_(read|write)_begin\(\) : \(\) -> \(\)",
            body,
        )
        if not match:
            raise SimError("unsupported runtime call: " + text)
        machine.ddr_policy = match.group(1)
        return
    if name.startswith("func.return") or name == "return":
        raise _ReturnSignal()
    if name.startswith("func.") or name == "module":
        return
    raise SimError("unsupported operation: " + text)


class _ReturnSignal(Exception):
    pass


def _contiguous_strides(sizes):
    strides = [1] * len(sizes)
    for index in range(len(sizes) - 2, -1, -1):
        strides[index] = strides[index + 1] * sizes[index + 1]
    return strides


def _iterate(sizes):
    if not sizes:
        yield []
        return
    for index in range(sizes[0]):
        for rest in _iterate(sizes[1:]):
            yield [index] + rest


def interpret_scf_for(op, machine, op_body):
    match = re.match(
        r"scf\.for (%\S+) = (%\S+) to (%\S+) step (%\S+)"
        r"(?: iter_args\((.*?)\) -> \((.*?)\))?",
        op_body,
    )
    if not match:
        raise SimError("unsupported scf.for: " + op.text)
    induction = match.group(1)
    lower = constant_of(machine, match.group(2))
    upper = constant_of(machine, match.group(3))
    step = constant_of(machine, match.group(4))
    iter_args = []
    if match.group(5):
        for piece in _split_top_level(match.group(5)):
            name, _, value = piece.partition("=")
            iter_args.append((name.strip(), value.strip()))
    results = result_name(op.text)
    values = []
    # The iter-arguments hold their initial values once, before the first
    # iteration; afterwards each iteration carries the previous `scf.yield`.
    for name, init in iter_args:
        machine.values[name] = machine.values[init]
    index = lower
    while index < upper:
        machine.values[induction] = index
        machine.yielded = None
        for child in op.blocks[0]:
            run_op(child, machine, None)
        yielded = machine.yielded or []
        if iter_args:
            if len(yielded) != len(iter_args):
                raise SimError("scf.yield arity does not match iter_args")
            for (name, _), value in zip(iter_args, yielded):
                machine.values[name] = value
            values = yielded
        index += step
    if results:
        # `%r:2 = scf.for ...`: the pass emits one result per iter_arg.
        names = [piece.strip() for piece in results.split(":")[0].split(",")]
        if len(names) == 1:
            machine.values[names[0]] = values[0] if values else None
        else:
            for name, value in zip(names, values):
                machine.values[name] = value


def interpret_scf_if(op, machine):
    condition = constant_of(machine, parse_operands(op.text, 0)[0])
    block = (
        op.blocks[0]
        if condition
        else (op.blocks[1] if len(op.blocks) > 1 else [])
    )
    machine.yielded = None
    for child in block:
        run_op(child, machine, None)
    yielded = machine.yielded or []
    results = result_name(op.text)
    if results:
        names = [piece.strip() for piece in results.split(":")[0].split(",")]
        if len(names) == 1:
            machine.values[names[0]] = yielded[0] if yielded else None
        else:
            for name, value in zip(names, yielded):
                machine.values[name] = value


def interpret_linalg_generic(op, machine, op_body):
    match = re.search(r"ins\((.*?)\) outs\((.*?)\)", op_body)
    if not match:
        raise SimError("unsupported linalg.generic: " + op.text)
    ins = [
        piece.split(":")[0].strip()
        for piece in _split_top_level(match.group(1))
    ]
    outs = [
        piece.split(":")[0].strip()
        for piece in _split_top_level(match.group(2))
    ]
    inputs = [value_of(machine, name) for name in ins]
    outputs = [value_of(machine, name) for name in outs]
    body_block = op.blocks[0]
    argument_names = []
    body_ops = []
    for child in body_block:
        if child.text.startswith("^bb0"):
            arguments = child.text[
                child.text.index("(") + 1 : child.text.index(")")
            ]
            argument_names = [
                piece.split(":")[0].strip() for piece in arguments.split(",")
            ]
            continue
        body_ops.append(child)
    for index in _iterate(outputs[0].sizes):
        for position, name in enumerate(argument_names):
            if position < len(inputs):
                machine.values[name] = inputs[position].load(index, machine)
            else:
                machine.values[name] = outputs[position - len(inputs)].load(
                    index, machine
                )
        machine.yielded = None
        for child in body_ops:
            run_op(child, machine, None)
        outputs[0].store(index, machine.yielded[0], machine)


def interpret_tile_op(op, machine, op_body, op_types):
    name = op_name(op.text)
    operands = parse_operands(op_body, 0)
    parsed = types_before_arrow(op_types)
    result = result_name(op.text)
    if name == "bosc_ame.tile.load_a" or name == "bosc_ame.tile.load_b":
        memref = value_of(machine, operands[0])
        shape = parse_result_type(op_types).shape
        matrix = [
            [memref.load([i, j], machine) for j in range(shape[1])]
            for i in range(shape[0])
        ]
        machine.values[result] = matrix
        return
    if name == "bosc_ame.tile.load_acc":
        memref = value_of(machine, operands[0])
        shape = parse_result_type(op_types).shape
        machine.values[result] = [
            [memref.load([i, j], machine) for j in range(shape[1])]
            for i in range(shape[0])
        ]
        return
    if name == "bosc_ame.tile.zero_acc":
        zero_type = parse_result_type(op_types)
        if not isinstance(zero_type, VectorType):
            raise SimError("zero_acc requires a 1..64 by 1..64 i32 result")
        shape = zero_type.shape
        if (
            operands
            or zero_type.element != "i32"
            or len(shape) != 2
            or any(not 1 <= size <= 64 for size in shape)
        ):
            raise SimError("zero_acc requires a 1..64 by 1..64 i32 result")
        if machine.outstanding:
            raise SimError(
                "zero_acc cannot start a matrix lifetime before DMA wait"
            )
        machine.values[result] = [[0] * shape[1] for _ in range(shape[0])]
        return
    if name == "bosc_ame.tile.mma":
        accumulator = value_of(machine, operands[0])
        lhs = value_of(machine, operands[1])
        rhs = value_of(machine, operands[2])
        rows, inner, columns = len(lhs), len(lhs[0]), len(rhs[0])
        matrix = [row[:] for row in accumulator]
        for i in range(rows):
            for j in range(columns):
                total = matrix[i][j]
                for k in range(inner):
                    total += lhs[i][k] * rhs[k][j]
                matrix[i][j] = _wrap_i32(total)
        machine.values[result] = matrix
        return
    if name == "bosc_ame.tile.store_acc":
        matrix = value_of(machine, operands[0])
        memref = value_of(machine, operands[1])
        if machine.outstanding:
            raise SimError("tile store before DMA wait")
        direct = "bosc_ame.direct_ddr" in op.text
        if direct and memref.spm_bank is not None:
            raise SimError("direct tile store must target external DDR")
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if direct:
                    memref.store_byte_address(
                        memref.byte_address([i, j]),
                        matrix[i][j],
                        machine,
                        direct=True,
                    )
                else:
                    memref.store([i, j], matrix[i][j], machine)
        return
    raise SimError("unsupported tile op: " + op.text)


def _wrap_i32(value):
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= (1 << 31) else value


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def simulate(module_path, function_name, buffers, scalars=None):
    """Run one function of an emitted BOSCAME module.

    `buffers` maps the DDR argument names (`%arg0`, ...) to `bytearray`s;
    `scalars` binds the program-information arguments (num_programs and
    program ids) so a specific program of the launch grid can be simulated.
    Returns the machine, whose `ddr` holds the resulting memory and whose
    `accesses`/`dma_log` hold every DDR access.
    """
    scalars = scalars or {}
    ops = read_module(module_path)
    machine = Machine()

    function = None
    for op in ops:
        if op_name(op.text) == "func.func" and op.blocks:
            match = re.search(r"@([A-Za-z0-9_]+)", op.text)
            if match and (
                function_name is None or match.group(1) == function_name
            ):
                function = op
                break
    if function is None:
        raise SimError("function not found: " + str(function_name))

    signature = function.text[function.text.index("(") + 1 :]
    signature = signature[: signature.index(")")]
    for piece in _split_top_level(signature):
        name, _, type_text = piece.partition(":")
        name = name.strip()
        type_text = type_text.strip()
        if type_text.startswith("memref<"):
            if name not in buffers:
                raise SimError("missing DDR buffer for " + name)
            data = buffers[name]
            if not isinstance(data, bytearray):
                data = bytearray(data)
                buffers[name] = data
            machine.ddr[name] = data
            machine.values[name] = MemRef(
                data, 0, [len(data)], [1], "i8", None, name
            )
        else:
            machine.values[name] = scalars.get(name, 0)

    for child in function.blocks[0]:
        try:
            run_op(child, machine, None)
        except _ReturnSignal:
            break
    return machine, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module")
    parser.add_argument("--function", default=None)
    parser.add_argument(
        "--buffers",
        required=True,
        help="JSON: {arg name: {'size': N, 'fill': byte}}",
    )
    parser.add_argument("--state")
    args = parser.parse_args()

    with open(args.buffers) as handle:
        spec = json.loads(handle.read())
    buffers = {}
    for name, description in spec.items():
        size = description["size"]
        fill = description.get("fill", 0)
        buffers[name] = bytearray([fill] * size)
    machine, _ = simulate(args.module, args.function, buffers)
    state = {
        "ddr": {name: list(data) for name, data in machine.ddr.items()},
        "accesses": machine.accesses,
        "dma": machine.dma_log,
    }
    with open(args.state, "w") as handle:
        json.dump(state, handle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
