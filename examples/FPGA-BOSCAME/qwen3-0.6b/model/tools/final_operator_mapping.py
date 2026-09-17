#!/usr/bin/env python3
"""Audit final replacement IR, linked adapters, NR ABI and per-kernel assembly.

No import/recompile of the model, old operator inventory, or FPGA operation.
Run with the Python matching build-python/python_packages (Buddy MLIR bindings).
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe_type(typ):
    from buddy_mlir import ir
    text = str(typ)
    if text.startswith('tensor<'):
        t = ir.RankedTensorType(typ)
        shape = list(t.shape)
        return {"type": text, "shape": shape, "dtype": str(t.element_type),
                "rank": len(shape), "logical_elements": math.prod(shape),
                "layout": "logical ranked tensor; physical layout is established after bufferization"}
    return {"type": text, "shape": None, "dtype": text}


def semantic_weight(tensor):
    layer = re.search(r"model.layers.(\d+)\.", tensor or "")
    suffixes = {"self_attn.q_proj.weight":"q_projection", "self_attn.k_proj.weight":"k_projection",
        "self_attn.v_proj.weight":"v_projection", "self_attn.o_proj.weight":"attention_output_projection",
        "mlp.gate_proj.weight":"gate_projection", "mlp.up_proj.weight":"up_projection",
        "mlp.down_proj.weight":"down_projection", "input_layernorm.weight":"input_rmsnorm",
        "post_attention_layernorm.weight":"post_attention_rmsnorm", "self_attn.q_norm.weight":"q_norm",
        "self_attn.k_norm.weight":"k_norm", "model.norm.weight":"final_rmsnorm",
        "lm_head.weight":"lm_head"}
    labels = [label for suffix,label in suffixes.items() if (tensor or "").endswith(suffix)]
    if len(labels) != 1:
        raise ValueError("unknown checkpoint weight role: " + str(tensor))
    return labels[0], int(layer[1]) if layer else None


def bindings(report, layout, sequence):
    """Join exact structural replacement records to value-verified checkpoint sections."""
    if layout.get('status') != 'PASS':
        raise ValueError('checkpoint layout has not passed verification')
    if len(report["original_parameters"]) != len(layout["sections"]):
        raise ValueError("original parameters and verified checkpoint layout disagree")
    params = dict(zip(report["original_parameters"], layout["sections"]))
    result = {}

    def put(symbol, record):
        section = record.get('checkpoint_weight')
        if section and section.get('match_kind') != 'value-match':
            raise ValueError('kernel weight is not value-verified: ' + symbol)
        if symbol in result and result[symbol] != record:
            raise ValueError("conflicting semantic records: " + symbol)
        result[symbol] = record

    for item in report["replaced"]:
        section = params.get(item["operands"][1]) if item["kind"] == "rmsnorm" else None
        role, layer = semantic_weight(section["checkpoint_tensor"]) if section else (item["kind"], None)
        put(item["symbol"], {"semantic": role, "layer": layer, "case": item["archive_entry"].removeprefix("_mlir_ciface_kernel_"),
            "source": "structurally matched replacement record plus value-verified checkpoint weight", "record": item,
            "checkpoint_weight": section, "write_arguments": [], "returns_output": True,
            "effect": "overwrite callsite-private static output; no input mutation; RMS scratch shared serially"})
    for item in report["w8a8_linears"]:
        section = params[item["weight_param"]]
        if section["shape"] != [item["n"], item["k"]]:
            raise ValueError("projection checkpoint dimensions disagree")
        role, layer = semantic_weight(section["checkpoint_tensor"])
        for stage, case, writes, effect in zip(("quantize","matmul","dequantize"), item["cases"],
                ([1,2],[2],[3]), ("overwrite int8 activation and per-token scale", "int32 C += A_i8 * W_i8^T; caller zeroes C every invocation", "overwrite f32 output; int32 accumulator * row scale * channel scale")):
            symbol = f"qwen_graph_w8a8_{item['node']}_{case}"
            put(symbol,{"semantic":role,"layer":layer,"stage":stage,"case":case,"record":item,
                "source":"structural Matmul(A, Permute(W,[1,0])) match joined to value-verified checkpoint weight",
                "checkpoint_weight":section,"write_arguments":writes,"returns_output":False,"effect":effect})
    for item in report["w8a8_embeddings"]:
        case = item["case"]
        put(f"qwen_graph_w8a8_{item['node']}_{case}", {"semantic":"embedding","layer":None,"case":case,
            "record":item,"checkpoint_weight":params[item["source_param"]],"write_arguments":[1],"returns_output":False,
            "source":"structural embedding indices/weight/output shape record; tied checkpoint weight",
            "effect":"overwrite fp32 gather = int8 tied embedding row * row scale"})
    attention = [x for x in report["attention_replacements"] if x["sequence"] == sequence]
    for item in attention:
        for stage, case, writes in zip(("qk","mask","softmax","pv"), item["cases"], ([3],[2],[1,2,3],[3])):
            put(f"qwen_graph_attn_{item['node']}_{stage}_{case}", {"semantic":"attention","stage":stage,"layer":None,
                "case":case,"record":item,"write_arguments":writes,"returns_output":False,
                "source":"Q/K/V shape+dtype, scale/dropout and runtime causal-mask structural proof",
                "effect":"overwrite destination/scratch; read cache only; valid=Position[S-1]+1; physical capacity 512"})
    for item in [x for x in report["kv_cache_replacements"] if x["sequence"] == sequence]:
        for stage, case, writes in (("layout",item["layout_case"],[1]),("update",item["case"],[2])):
            put(f"qwen_graph_kv_{item['node']}_{case}", {"semantic":"kv_cache","stage":stage,"layer":None,
                "case":case,"record":item,"write_arguments":writes,"returns_output":False,
                "source":"IndexPut dimension=2 overwrite with proven position+arange(S) and cache layout",
                "effect":"overwrite active cache slots in-place; untouched slots persist" if stage=="update" else "overwrite layout workspace [S,8,128] from [1,8,S,128]"})
    return result


def fallback(op, shapes, result_shapes, sequence):
    name = op.operation.name
    target = result_shapes[0] if result_shapes else None
    if name in ("arith.constant","tosa.const","tosa.const_shape","tensor.empty"):
        return "constant_or_storage", "support"
    if name == "tosa.matmul":
        if shapes[:2] != [[1,64,1],[1,1,sequence]]:
            return "uncovered_dense_matmul", "compute"
        return "rope_frequency_outer_product", "compute"
    if name in ("math.sin","math.cos"):
        return "rope_trigonometric_table", "compute"
    if name == "tosa.add" and target == [1,8,2,512,128] and [1,8,1,512,128] in shapes:
        # The second operand is the explicit zero tensor inserted for broadcast.
        zero = op.operands[1].owner
        if getattr(zero,"name",None) != "tosa.const" or "dense<0.000000e+00>" not in zero.get_asm(use_local_scope=True,assume_verified=True):
            raise ValueError("GQA repeat is not the expected zero broadcast")
        return "gqa_expand_full_cache", "memory"
    if name == "tosa.add" and shapes[:2] == [[1,sequence,1024]]*2:
        return "residual_add", "compute"
    if name == "tosa.mul" and shapes[:2] == [[1,sequence,3072]]*2:
        return "swiglu_gate_multiply", "compute"
    if name == "tosa.mul" and target and len(target)==4 and target[-1]==128 and [1,1,sequence,128] in shapes:
        return "rope_broadcast_multiply", "compute"
    if name == "tosa.add" and target in ([1,16,sequence,128],[1,8,sequence,128]) and shapes[0]==shapes[1]:
        return "rope_rotated_sum", "compute"
    if name == "tensor.extract_slice":
        return ("rope_half_slice" if target and target[-1]==64 else "lm_head_last_token_slice"), "view"
    if name == "tensor.insert_slice":
        return "rope_rotated_half_join", "memory"
    if name == "tosa.negate":
        return "rope_negated_half", "compute"
    if name == "tosa.add" and target == [1,sequence,2,64]:
        return "rope_frequency_duplicate", "memory"
    if name == "tosa.mul" and target == [1,sequence,128]:
        return "rope_table_scale", "compute"
    if name == "tosa.transpose":
        return "layout_transpose", "memory"
    if name == "tosa.reshape":
        return "reshape_view", "view"
    if name == "linalg.generic" and not any(list(x.uses) for x in op.results):
        return "unused_original_causal_mask", "dead"
    if name in ("tensor.generate","tosa.greater_equal","tosa.cast") or (name=="tosa.add" and any("i64" in str(v.type) for v in op.operands)):
        return "cache_position_or_mask_index_math", "compute"
    if name=="func.return":
        return "entry_return", "control"
    return "unclassified_"+name, "compute"


def kernel_index(library):
    archive = load(library/"archive.json")
    records = {}
    ame_prefixes=("mset","mlae","mlbe","mlce","msce","mqma","mrelease","mzero")
    for entry in archive["cases"]:
        name=entry["case"];d=library/"evidence"/name;assembly=d/"kernel.s"
        verified_files = {}
        for filename, digest in entry['files_sha256'].items():
            path = library / (name + '.' + filename) if filename in ('kernel.o', 'adapter.o') else d / filename
            if sha(path) != digest:
                raise ValueError('archived kernel evidence changed: ' + str(path))
            verified_files[str(path)] = digest
        instructions=Counter(re.findall(r"^\s+([a-z][a-z0-9_.]*)\s",assembly.read_text(),re.M))
        vectors={n:c for n,c in instructions.items() if n.startswith("v")}
        arithmetic={n:c for n,c in vectors.items() if not n.startswith(("vset","vle","vse","vlse","vsse","vmv"))}
        ame={n:c for n,c in instructions.items() if n.startswith(ame_prefixes)}
        record={"case":name,"frontend":entry["frontend"],"archive_symbols":entry["symbols"],
            "assembly":str(assembly),"assembly_sha256":sha(assembly),"llvm":str(d/"kernel.ll"),
            "llvm_sha256":sha(d/"kernel.ll"),"ttir":str(d/"kernel.ttir"),"ttir_sha256":sha(d/"kernel.ttir"),
            "linalg":str(d/"kernel.linalg.mlir"),"linalg_sha256":sha(d/"kernel.linalg.mlir"),
            "verified_archive_files_sha256":verified_files,
            "static_instruction_counts":{"ame":ame,"rvv":vectors,"rvv_arithmetic":arithmetic},
            "classification":"AME int8 matrix arithmetic" if 'mqma.b.mm' in ame else "RVV floating-point dot" if 'vfmacc.vf' in arithmetic else "scalar arithmetic with RVV memory movement" if vectors else "scalar instructions",
            "scope":"kernel.s only, not the whole ELF; static instruction sites are not executed instruction counts; no claim that generic Buddy fallback uses this ISA"}
        records[name]=record
    return records


def actual_descriptors(llvm, entry_abi):
    """Resolve constant fields; retain explicit provenance for runtime fields.

    This is a narrow parser for this compiler's unoptimized LLVM. It never
    substitutes logical tensor shapes for an unknown physical descriptor.
    """
    entry=entry_abi['entry']
    match=re.search(r'^define [^\n]*@'+re.escape(entry)+r'\(([^\n]*)\) \{\n(.*?)^\}',llvm,re.M|re.S)
    if not match:raise ValueError('missing actual LLVM entry '+entry)
    args=[part.strip().split()[-1] for part in match[1].split(',')]
    labels=[]
    for item in entry_abi['inputs']:
        n=item['node'];rank=item['rank']
        labels += [f'entry.{n}.allocated',f'entry.{n}.aligned',f'entry.{n}.offset']
        labels += [f'entry.{n}.size[{i}]' for i in range(rank)]
        labels += [f'entry.{n}.stride[{i}]' for i in range(rank)]
    if len(labels)!=len(args):raise ValueError('entry ABI mismatch')
    arguments=dict(zip(args,labels));definitions=dict(re.findall(r'^\s*(%[\w.]+) = (.*)$',match[2],re.M))
    memo={}
    def evaluate(value,field=(),depth=0):
        key=(value,field)
        if key in memo:return memo[key]
        if depth>120:return {'unresolved':value,'field':list(field),'reason':'depth limit'}
        if not field and re.fullmatch(r'-?\d+',value):return int(value)
        if value in arguments:return {'runtime':arguments[value],'field':list(field)}
        expression=definitions.get(value,'')
        inserted=re.fullmatch(r'insertvalue \{.*\} (%[\w.]+|poison), (?:i64|ptr) ([^, ]+), ([0-9, ]+)',expression)
        extracted=re.fullmatch(r'extractvalue \{.*\} (%[\w.]+), ([0-9, ]+)',expression)
        if inserted:
            index=tuple(map(int,inserted[3].split(',')))
            answer=evaluate(inserted[2],(),depth+1) if index==field else evaluate(inserted[1],field,depth+1)
        elif extracted:
            answer=evaluate(extracted[1],tuple(map(int,extracted[2].split(',')))+field,depth+1)
        elif expression.startswith('call '):
            callee=re.search(r'@([^ (]+)',expression)
            answer={'returned_descriptor':callee[1] if callee else expression[:160],'field':list(field)}
        elif not field and re.match(r'(?:add|mul|sub) (?:nuw |nsw )*i64 ',expression):
            binary=re.fullmatch(r'(add|mul|sub) (?:nuw |nsw )*i64 ([^,]+), (.+)',expression)
            left=evaluate(binary[2],(),depth+1);right=evaluate(binary[3],(),depth+1)
            answer=({'add':lambda a,b:a+b,'mul':lambda a,b:a*b,'sub':lambda a,b:a-b}[binary[1]](left,right)
                    if isinstance(left,int) and isinstance(right,int) else {'expression':binary[1],'left':left,'right':right})
        else:answer={'unresolved':value,'field':list(field),'definition':expression[:220]}
        memo[key]=answer;return answer
    calls={}
    for call in re.finditer(r'\bcall [^\n]*?@(qwen_graph_[^ (]+)\(([^\n]*)\)',match[2]):
        if call[1] in calls:raise ValueError('duplicate callsite symbol in actual entry')
        calls[call[1]]=[part.strip().split()[-1] for part in call[2].split(',')]
    def descriptors(symbol,ranks):
        values=calls[symbol];cursor=0;result=[]
        for rank in ranks:
            raw=values[cursor:cursor+3+2*rank];cursor+=len(raw)
            result.append({'rank':rank,'actual_llvm_fields':raw,
                'offset_elements':evaluate(raw[2]),'sizes':[evaluate(x) for x in raw[3:3+rank]],
                'strides':[evaluate(x) for x in raw[3+rank:]]})
        if cursor!=len(values):raise ValueError('flattened external ABI count mismatch '+symbol)
        return result
    runtime_calls = Counter(re.findall(r'\bcall [^\n]*?@((?:llvm\.mem[^ (]*|malloc|free|memcpy|memmove|memset|aligned_alloc|realloc))\(',match[2]))
    return descriptors, {'static_call_sites':dict(runtime_calls),
        'scope':'actual unoptimized forward entry LLVM body; sites inside loops may execute many times; this is not traffic, peak memory, or cycle measurement'}


def module_report(path, nr, binding, kernels, expected, sequence):
    from buddy_mlir import ir
    llvm=(nr/f'forward_{"prefill" if sequence>1 else "decode"}.ll').read_text()
    entry_abi=load(nr/'entry-abi.json')
    descriptors,runtime_calls=actual_descriptors(llvm,entry_abi)
    raw_defs=dict(re.findall(r"^(define private [^\n]*@([^ (]+)\([^\n]*\) \{)$",llvm,re.M))
    # Normalize by symbol; the exact private trampoline definition is retained.
    raw_defs={symbol:line for line,symbol in raw_defs.items()}
    pointer_decls={s:line for line,s in re.findall(r"^(declare [^\n]*@(_mlir_ciface_[^ (]+)\([^\n]*\))$",llvm,re.M)}
    with ir.Context() as context:
        context.enable_multithreading(False)
        context.allow_unregistered_dialects=True
        module=ir.Module.parse(path.read_text());function=next(op for op in module.body.operations if op.operation.name=='func.func' and len(op.regions[0].blocks))
        block=function.regions[0].blocks[0];ops=list(block.operations)
        names={value:f'arg{index}' for index,value in enumerate(block.arguments)}
        origins={};records=[];calls=[];fallbacks=[]
        for index,op in enumerate(ops):
            inputs=[describe_type(v.type) for v in op.operands];outputs=[describe_type(v.type) for v in op.results]
            for j,value in enumerate(op.results):names[value]=f'op{index}.r{j}'
            upstream=sorted({p for value in op.operands for p in origins.get(value,())})
            record={"index":index,"operation":op.operation.name,"operands":[names.get(v,'nested_value') for v in op.operands],
                "input_types":inputs,"results":[names[v] for v in op.results],"output_types":outputs,
                "upstream_external_calls":upstream,"ir":op.operation.get_asm(use_local_scope=True,assume_verified=True,skip_regions=True).splitlines()[0],
                "attributes":{key:str(op.attributes[key]) for key in op.attributes}}
            if op.operation.name=='func.call':
                symbol=str(op.attributes['callee']).removeprefix('@')
                if symbol not in binding:raise ValueError('no structural semantic provenance for '+symbol)
                source=binding[symbol];case=source['case'];kernel=kernels[case]
                if symbol not in raw_defs or '_mlir_ciface_'+symbol not in pointer_decls:
                    raise ValueError('external call absent from NR LLVM ABI: '+symbol)
                record.update({k:v for k,v in source.items() if k!='record'})
                record['structural_match_record']=source['record']
                record['symbol']=symbol;record['c_symbol']='_mlir_ciface_'+symbol
                record['kernel_c_symbol']='_mlir_ciface_kernel_'+case
                record['triton_symbol']=kernel['frontend']['symbol']
                if not {record['kernel_c_symbol'],record['triton_symbol']} <= set(kernel['archive_symbols']):
                    raise ValueError('missing kernel/adapter archive symbol: '+symbol)
                record['bufferized_abi']={'raw_private_trampoline':raw_defs[symbol],'c_declaration':pointer_decls['_mlir_ciface_'+symbol],
                    'descriptor_layout':'allocated pointer, aligned pointer, int64 offset, int64 sizes[rank], int64 strides[rank]',
                    'call_convention':'one descriptor pointer per argument; returning kernels receive uninitialized result descriptor pointer first',
                    'kernel_adapter':'extract aligned + offset*sizeof(dtype), then static physical indexing; arbitrary strides unsupported',
                    'graph_argument_ranks':[v.get('rank') for v in inputs],
                    'kernel_argument_ranks':[v['rank'] for v in kernel['frontend']['arguments']]}
                record['bufferized_abi']['actual_argument_descriptors']=descriptors(symbol,[v['rank'] for v in inputs])
                record['offset_alias_lifetime']={'offset':'descriptor offset honored by kernel adapter',
                    'alias':'KV destination aliases persistent caller cache; other output workspaces distinct caller-owned buffers' if source['semantic']=='kv_cache' else 'returning output is static per callsite; void output is caller-owned; no intentional input/output alias except int32 accumulating C',
                    'lifetime':'caller workspaces survive prefill/decode; static returning buffers survive call and are non-reentrant; no malloc in kernel archive'}
                record['kernel_isa']=kernel['classification']
                record['source_dataflow_layer_candidates']=sorted({calls[p]['layer'] for p in upstream if calls[p].get('layer') is not None})
                if record.get('layer') is None and len(record['source_dataflow_layer_candidates'])==1:
                    record['layer']=record['source_dataflow_layer_candidates'][0]
                call_index=len(calls);record['call_order']=call_index;calls.append(record)
                for argument in source['write_arguments']:origins[op.operands[argument]]={call_index}
                for value in op.results:origins[value]={call_index}
            else:
                category,role=fallback(op,[x['shape'] for x in inputs],[x['shape'] for x in outputs],sequence)
                record['semantic']=category;record['role']=role
                record['implementation']='generic Buddy lowering; no Triton external kernel'
                record['bufferization']={'stride_offset':'internal descriptor/allocation choices belong to final LLVM; tensor IR alone does not establish a physical view',
                    'alias_inplace':'reshape/slice may alias or materialize; transpose/broadcast/join can copy; no blanket zero-copy claim',
                    'accumulation':'overwrite/new logical tensor; only frequency matmul reduces products'}
                record['performance']='not timed per fallback; generic LLVM ISA is not inferred from kernel or whole-ELF RVV counts'
                if role=='memory':record['logical_output_bytes_f32']=sum(x.get('logical_elements',0)*4 for x in outputs if x['dtype']=='f32')
                if category=='gqa_expand_full_cache':record['performance']='full 512-capacity K/V expansion: 4 MiB output each even when valid length is small; memory cost remains'
                if op.operation.name=='tosa.matmul':
                    a,b=inputs[0]['shape'],inputs[1]['shape'];macs=math.prod(a[:-2])*a[-2]*a[-1]*b[-1]
                    record['estimated_macs']=macs;record['large_uncovered']=macs>=1_000_000
                fallbacks.append(record)
                for value in op.results:origins[value]=set(upstream)
            records.append(record)
        if len(calls)!=expected:raise ValueError('IR call count disagrees with final replacement report')
        return {'source':str(path),'sha256':sha(path),'top_level_operation_count':len(ops),'external_call_count':len(calls),
            'external_call_semantics':dict(Counter(x['semantic'] for x in calls)),
            'fallback_counts':dict(Counter(x['semantic'] for x in fallbacks)),
            'operations':records,'entry_abi':entry_abi,'native_key_abi':load(nr/'native-key-abi.json'),
            'runtime_memory_call_sites':runtime_calls,
            'dense_coverage':load(nr/'nr-lower.json')['dense_coverage'],
            'llvm_sha256':sha(nr/f'forward_{"prefill" if sequence>1 else "decode"}.ll')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--layout',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=load(args.build/'replacement/triton-call-replacement.json');layout=load(args.layout)
    kernels=kernel_index(args.build/'model-lib');graphs={}
    for kind,sequence in (('prefill',16),('decode',1)):
        graphs[kind]=module_report(args.build/f'replacement/subgraph0_{kind}.triton.mlir',args.build/f'nr-{kind}',
            bindings(report,layout,sequence),kernels,report['graphs'][kind]['external_calls'],sequence)
    adapters=(args.build/'replacement/qwen_triton_adapters.c').read_text()
    for graph in graphs.values():
        for op in graph['operations']:
            if op['operation']=='func.call':
                signature=re.search(r'^void '+re.escape(op['c_symbol'])+r'\([^\n]*\) \{\n(.*?)^}',adapters,re.M|re.S)
                if not signature:raise ValueError('missing graph-facing adapter '+op['c_symbol'])
                if not re.search(r'\b'+re.escape(op['kernel_c_symbol'])+r'\(',signature[1]):
                    raise ValueError('graph adapter does not call recorded Triton kernel: '+op['c_symbol'])
                op['adapter_c_signature']=signature[0].splitlines()[0]
    unclassified=[{'entry':kind,'index':op['index'],'semantic':op['semantic']} for kind,g in graphs.items() for op in g['operations'] if op['semantic'].startswith('unclassified_')]
    large=[{'entry':kind,**item} for kind,g in graphs.items() for item in g['dense_coverage']['large_uncovered']]
    args.output.mkdir(parents=True,exist_ok=True)
    mapping={'status':'PASS' if not unclassified and not large else 'PARTIAL','status_scope':'operator classification and artifact/ABI provenance only; this is not full-model FPGA numerical validation','variant':'native-K 28-layer final artifacts','method':'actual replacement subgraph operation/dataflow traversal + structural replacement records + value-verified checkpoint layout + actual NR LLVM ABI + linked adapter signatures',
        'sources':{str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('quant_reference.py'),args.layout,args.build/'replacement/triton-call-replacement.json',args.build/'replacement/qwen_triton_adapters.c',args.build/'model-lib/archive.json',args.build/'model-lib/libqwen_triton.a',args.build/'model-lib/external-link.json')},
        'quantization_contract':{
            'weight':'per output row of physical W[N,K]; signed symmetric int8 [-127,127]; FP32 scale=max(abs(row))/127, zero row scale=1',
            'activation':'per token/row of A[M,K], same scale and signed range as weight; FP32 division then sign-dependent +/-0.5 and truncation, saturating [-127,127]',
            'zero_point':0,'accumulator':'signed int32 C += A_i8 * W_i8^T, explicitly reset to zero before each graph invocation',
            'dequantization':'(float32(accumulator) * activation_row_scale) * weight_channel_scale; preserve FP32 operation order',
            'bias':'none','residual_and_norm':'FP32 graph path; no int8 residual accumulator',
            'embedding_and_lm_head':'one tied quantized matrix and row-scale table; embedding gathers then multiplies by scale; lm_head uses W8A8 linear',
            'reference':'tools/quant_reference.py (SHA256 in sources); independent numerical implementation, not graph forward ordering'},
        'large_uncovered_criterion':'unreplaced matmul/attention with unknown estimated MACs or >=1,000,000 MACs; memory copies and elementwise scalar work are separately reported, not hidden by this criterion',
        'unclassified':unclassified,'large_uncovered':large,'graphs':graphs,
        'limitations':['Structured replacement IR uses a GraphDriver subgraph ABI; deployed direct-entry ABI is separately taken from entry-abi.json and final LLVM','An operation present but unused in structured IR is not necessarily executed after canonicalization','Internal generic fallback buffer alias/lifetime and RVV instruction selection are not inferred from logical tensor shapes','Per-kernel scalar math may coexist with RVV memory movement; only kernel-local assembly justifies ISA classification']}
    (args.output/'operator-mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    (args.output/'kernel-isa.json').write_text(json.dumps({'scope':'archived kernel-local assembly, not entire ELF','kernels':kernels},indent=2)+'\n')
    lines=['# Final native-K model operator coverage','',f'Mapping status: {mapping["status"]}. Actual final graph: prefill {graphs["prefill"]["external_call_count"]} calls; decode {graphs["decode"]["external_call_count"]} calls. This status covers operator classification and artifact/ABI provenance; it does not assert full-model FPGA numerical validation.','',
        'Sources and SHA256 values, every IR operation, semantic evidence, checkpoint weight, shape/dtype, C/LLVM ABI and call order are in `operator-mapping.json`. Per-kernel AME/RVV instructions are in `kernel-isa.json`.','',
        '| Semantic | Prefill operations | Decode operations |','|---|---:|---:|']
    semantics=sorted(set(graphs['prefill']['external_call_semantics'])|set(graphs['decode']['external_call_semantics']))
    for semantic in semantics:lines.append('| '+semantic+' | '+str(graphs['prefill']['external_call_semantics'].get(semantic,0))+' | '+str(graphs['decode']['external_call_semantics'].get(semantic,0))+' |')
    lines+=['','Each projection has separate quantize, AME matmul and dequantize calls. Attention has QK, scale/mask, softmax and PV. KV includes layout and in-place cache writes.','',
        '## Remaining Buddy operations','', '| Operation group | Prefill | Decode |','|---|---:|---:|']
    for semantic in sorted(set(graphs['prefill']['fallback_counts'])|set(graphs['decode']['fallback_counts'])):
        lines.append('| '+semantic+' | '+str(graphs['prefill']['fallback_counts'].get(semantic,0))+' | '+str(graphs['decode']['fallback_counts'].get(semantic,0))+' |')
    lines+=['','No unclassified operations or large unreplaced dense operations are claimed only when the JSON status is PASS. The dense threshold is 1,000,000 MACs; it does not exempt memory movement from reporting.',
        '','GQA still expands both K and V over all 512 slots (4 MiB each, 56 expansions per entry = 224 MiB of logical output writes). Native QK removes a separate 4 MiB key transpose per layer; it does not remove GQA expansion.',
        '','RoPE frequency outer product is 1,024 MACs for prefill / 64 MACs for decode. Sin/cos, split-half rotations, residual adds, SwiGLU multiply, attention output transpose and last-token slice remain in Buddy. Their per-op ISA and timing have not been established by seeing RVV elsewhere in the ELF.',
        '','## Kernel-local ISA','', '| Kernel classification | Cases |','|---|---:|']
    for name,count in Counter(x['classification'] for x in kernels.values()).items():lines.append(f'| {name} | {count} |')
    lines+=['','Scalar arithmetic with RVV memory movement is explicitly distinct from RVV arithmetic. The archive contains kernel/ABI objects only; runtime, launch/test data and graph generic fallback are excluded.','',
        'Actual per-call descriptor offset/sizes/strides are resolved from final LLVM. Runtime fields retain entry-argument provenance, and unknown fields remain explicit. Logical tensor shapes are never substituted for unknown physical descriptors. The top-level JSON also states the exact W8A8 contract.','',
        'Reproduce from the repository root with the Python 3.11 environment matching the Buddy bindings:', '```bash',f'PYTHONPATH=build-python/python_packages python3 -B examples/FPGA-BOSCAME/qwen3-0.6b/model/tools/final_operator_mapping.py --build {args.build} --layout {args.layout} --output {args.output}','```','']
    (args.output/'coverage.md').write_text('\n'.join(lines))
    print(json.dumps({'status':mapping['status'],'calls':{k:g['external_call_count'] for k,g in graphs.items()},'unclassified':unclassified,'large_uncovered':large,'kernels':len(kernels)},indent=2))
    if mapping['status']!='PASS':return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
