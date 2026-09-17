#!/usr/bin/env python3
"""Host-test the actual portable C decoder/template against official resources.

No board access. Host tokenizer use here is exclusively a numerical reference;
this script is not part of the FPGA inference/interactive data path.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import random
import shlex
import subprocess
import struct
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from pack_tokenizer import pack


class Resource(C.Structure):
    _fields_=[("blob",C.c_void_p),("size",C.c_size_t),("slots",C.c_uint32),
              ("records",C.c_uint32),("pieces",C.c_uint32),("piece_size",C.c_uint32),
              ("merge_offset",C.c_uint32),("merge_count",C.c_uint32),
              ("added_offset",C.c_uint32),("added_count",C.c_uint32)]


class Decoder(C.Structure):
    _fields_=[("pending",C.c_uint8*4),("count",C.c_uint8),("expected",C.c_uint8)]


def main():
    from tokenizers import Tokenizer,pre_tokenizers
    from transformers import AutoTokenizer
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",type=Path,required=True)
    parser.add_argument("--build",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cc",default="cc")
    args=parser.parse_args()
    args.build.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    blob=args.build/"tokenizer.bin"
    resources=pack(args.assets,blob)
    library=args.build/"libqwen_text_check.so"
    subprocess.run([*shlex.split(args.cc),"-std=c11","-O2","-Wall","-Wextra","-Werror",
                    "-shared","-fPIC",str(root/"text/tokenizer_resource.c"),"-o",str(library)],check=True)
    lib=C.CDLL(str(library.resolve()))
    callback_type=C.CFUNCTYPE(None,C.c_void_p,C.POINTER(C.c_uint8),C.c_size_t)
    lib.qwen_tokenizer_open.argtypes=[C.POINTER(Resource),C.c_void_p,C.c_size_t]
    lib.qwen_token_piece.argtypes=[C.POINTER(Resource),C.c_uint32,C.POINTER(C.c_void_p),C.POINTER(C.c_size_t),C.POINTER(C.c_int)]
    lib.qwen_decode_token.argtypes=[C.POINTER(Resource),C.POINTER(Decoder),C.c_uint32,C.c_int,callback_type,C.c_void_p]
    lib.qwen_decode_bytes.argtypes=[C.POINTER(Decoder),C.c_void_p,C.c_size_t,callback_type,C.c_void_p]
    lib.qwen_decode_finish.argtypes=[C.POINTER(Decoder),callback_type,C.c_void_p]
    lib.qwen_chat_single_turn.argtypes=[C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t),C.c_void_p,C.c_size_t,C.c_int,C.c_void_p,C.c_size_t,C.c_int]
    raw=blob.read_bytes(); memory=C.create_string_buffer(raw); resource=Resource()
    assert lib.qwen_tokenizer_open(C.byref(resource),memory,len(raw))==0
    official=Tokenizer.from_file(str(args.assets/"tokenizer.json"))
    hf=AutoTokenizer.from_pretrained(str(args.assets),local_files_only=True)
    chunks=[]
    @callback_type
    def emit(_,data,size):chunks.append(C.string_at(data,size))
    def decode(ids,skip=False):
        chunks.clear(); state=Decoder()
        for token in ids:
            assert lib.qwen_decode_token(C.byref(resource),C.byref(state),token,int(skip),emit,None)==0
        lib.qwen_decode_finish(C.byref(state),emit,None)
        # Every callback is a complete UTF-8 code point, even when tokens split it.
        for chunk in chunks:chunk.decode("utf-8",errors="strict")
        return b"".join(chunks).decode()
    fixed=["Hello, world!","你好，世界！","I'm testing 12345.67\nNext line.",
           "e\u0301 café Å\u212b", "👩‍💻🚀🙂", " \t \r\n  end  ",
           "<|im_start|>user\n你好<|im_end|>\n", "<think>\n\n</think>\n\n",
           "a\x00b", "مرحبا بالعالم", "नमस्ते दुनिया", "한글 日本語"]
    fixtures=[];token_checks=0
    for text in fixed:
        ids=official.encode(text,add_special_tokens=False).ids
        for skip in (False,True):
            assert decode(ids,skip)==official.decode(ids,skip_special_tokens=skip)
            token_checks+=1
        fixtures.append({"text":text,"host_token_ids":ids,"fpga_token_ids":None})
    for added in resources["special_tokens"]:
        for skip in (False,True):
            assert decode([added["id"]],skip)==official.decode([added["id"]],skip_special_tokens=skip)
            token_checks+=1
    rng=random.Random(20260916)
    for _ in range(1000):
        ids=[rng.randrange(151669) for _ in range(rng.randrange(1,65))]
        assert decode(ids)==official.decode(ids,skip_special_tokens=False)
        token_checks+=1
    # Exhaust all one-byte cases, then random malformed/truncated multibyte data.
    byte_cases=[bytes([n]) for n in range(256)]
    byte_cases += [bytes(rng.randrange(256) for _ in range(rng.randrange(1,129))) for _ in range(1000)]
    for raw_bytes in byte_cases:
        chunks.clear();state=Decoder()
        for index in range(0,len(raw_bytes),3):
            part=raw_bytes[index:index+3]
            lib.qwen_decode_bytes(C.byref(state),C.c_char_p(part),len(part),emit,None)
        lib.qwen_decode_finish(C.byref(state),emit,None)
        assert b"".join(chunks).decode()==raw_bytes.decode("utf-8",errors="replace")
    # Compare actual C template rendering with official Jinja implementation.
    template_checks=0
    for system in (None,"","You are a helpful assistant."):
        for text in fixed:
            for thinking in (False,True):
                messages=[] if system is None else [{"role":"system","content":system}]
                messages.append({"role":"user","content":text})
                expected=hf.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=thinking).encode()
                buf=C.create_string_buffer(len(expected));size=C.c_size_t()
                sysbytes=(system or "").encode();usrbytes=text.encode()
                assert lib.qwen_chat_single_turn(buf,len(expected),C.byref(size),C.c_char_p(sysbytes),len(sysbytes),int(system is not None),C.c_char_p(usrbytes),len(usrbytes),int(thinking))==0
                assert bytes(buf)[:size.value]==expected
                assert lib.qwen_chat_single_turn(buf,len(expected)-1,C.byref(size),C.c_char_p(sysbytes),len(sysbytes),int(system is not None),C.c_char_p(usrbytes),len(usrbytes),int(thinking))==-1
                template_checks+=1
    # Truncated/corrupt headers and invalid/unassigned IDs must fail explicitly.
    rejected=0
    for truncated in (0,1,63,64,len(raw)-1):
        assert lib.qwen_tokenizer_open(C.byref(Resource()),memory,truncated)==-1;rejected+=1
    for offset in (0,8,12,16,32,36,40,44,48,52,56,60):
        broken=bytearray(raw);broken[offset:offset+4]=b"\xff"*4
        assert lib.qwen_tokenizer_open(C.byref(Resource()),C.create_string_buffer(bytes(broken)),len(raw))==-1;rejected+=1
    merge_offset,added_offset,byte_offset=struct.unpack_from("<III",raw,44)
    for offset in (merge_offset,merge_offset+4,merge_offset+8,merge_offset+12,
                   added_offset,byte_offset):
        broken=bytearray(raw);broken[offset:offset+4]=b"\xff"*4
        assert lib.qwen_tokenizer_open(C.byref(Resource()),C.create_string_buffer(bytes(broken)),len(raw))==-1;rejected+=1
    # Binary search requires strictly sorted unique merge keys, and a byte
    # alphabet entry must actually decode to that byte.
    broken=bytearray(raw);broken[merge_offset+16:merge_offset+24]=broken[merge_offset:merge_offset+8]
    assert lib.qwen_tokenizer_open(C.byref(Resource()),C.create_string_buffer(bytes(broken)),len(raw))==-1;rejected+=1
    broken=bytearray(raw);broken[byte_offset:byte_offset+4]=broken[byte_offset+4:byte_offset+8]
    assert lib.qwen_tokenizer_open(C.byref(Resource()),C.create_string_buffer(bytes(broken)),len(raw))==-1;rejected+=1
    pointer=C.c_void_p();length=C.c_size_t();special=C.c_int()
    for token in (151669,151935,151936,0xffffffff):
        assert lib.qwen_token_piece(C.byref(resource),token,C.byref(pointer),C.byref(length),C.byref(special))==-1;rejected+=1
    # Reproduce reference branch's space-only segmentation to expose differences.
    legacy=Tokenizer.from_file(str(args.assets/"tokenizer.json"))
    legacy.normalizer=None
    legacy.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False)
    differences=[]
    for text in fixed[:6]:
        parts=[];i=0
        while i<len(text):
            start=i
            if text[i]==" ":
                while i<len(text) and text[i]==" ":i+=1
            while i<len(text) and text[i]!=" ":i+=1
            parts.extend(legacy.encode(text[start:i],add_special_tokens=False).ids)
        actual=official.encode(text,add_special_tokens=False).ids
        if parts!=actual:differences.append({"text":text,"official":actual,"reference_space_split_equivalent":parts})
    assert differences,"compatibility audit should exercise the known mismatch"
    report={"status":"PASS","execution":"host: portable C; no FPGA or model inference",
            "token_sequence_decoder_checks":token_checks,"malformed_utf8_checks":len(byte_cases),
            "official_chat_template_checks":template_checks,"rejected_invalid_resources_or_ids":rejected,
            "tokenizer_blob_sha256":hashlib.sha256(raw).hexdigest(),"fixtures":fixtures,
            "reference_tokenizer_incompatibilities":differences,
            "encoder_status":"tested separately by check_tokenizer_encode.py; no FPGA token IDs in this host report",
            "sources":{str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (root/"text/tokenizer_resource.c",root/"text/tokenizer_resource.h",root/"tools/check_text.py")},
            "supported_chat_subset":"optional system + one user; no tools/history; thinking flag"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(f"PASS: {token_checks} decoder sequences, {len(byte_cases)} UTF-8, {template_checks} templates, {rejected} rejection cases")


if __name__=="__main__":main()
