import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import LlamaModel, LlamaConfig
import torch._dynamo as dynamo
from buddy.LlamaCompiler import DynamoCompiler

tokenizer = LlamaTokenizer.from_pretrained('/home/wlq/torch-mlir/examples/llama-hf/llama-2-7B-hf')
model = LlamaForCausalLM.from_pretrained('/home/wlq/torch-mlir/examples/llama-hf/llama-2-7B-hf', torchscript=True)
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
# generate_ids = model.generate(inputs.input_ids, max_length=30)
# print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
# print(len(generate_ids))
# print(generate_ids)
# print("-----------------------")
inputs = inputs.input_ids
print(inputs)
model.eval()
# scripted_module = torch.jit.trace(model, inputs)
# graph = scripted_module.graph.copy()
# torch._C._jit_pass_inline(graph)
# print(graph)
model_opt = dynamo.optimize(DynamoCompiler)(model)
# for (name, module) in model_opt.named_modules():
#     print(name, module)
# features_in_hook = []
# features_out_hook = []

# def hook(module, fea_in, fea_out):
#     features_in_hook.append(fea_in)
#     features_out_hook.append(fea_out)
#     return None

# layer_name = '_orig_mod.model.layers.0.self_attn.v_proj'
# for (name, module) in model_opt.named_modules():
#     if name == layer_name:
#         module.register_forward_hook(hook=hook)
result = model_opt(inputs)
# print("------------------")
# print(features_in_hook)
# print(features_out_hook)
# print("------------------")
print(result[0].shape)
print(result[0])
exit()
getattr_nodes = graph.findAllNodes("prim::GetAttr", recurse=True)

import itertools

def _get_uses(node):
    uses = []
    for output in node.outputs():
        uses += output.uses()
    return uses

def _get_users(node):
    return [use.user for use in _get_uses(node)]

def get_use_chains(root_node, terminate=lambda _: False):
    """
    Track a chain of users of this node forward, returning a list of chains
    See get_attr_chains below for its usage
    """

    def concat_lists(lists):
        return itertools.chain.from_iterable(lists)
    
    def inner(current, accum):
        users = _get_users(current)

        if not users or terminate(users):
            return [accum]
        
        return concat_lists([inner(nxt, accum+[nxt]) for nxt in users])
    
    return inner(root_node, [root_node])

def get_attr_chains(root_getattr_node):

    def terminate(users):
        next_attrs = [user for user in users if user.kind() == "prim::GetAttr"]
        return len(next_attrs) == 0
    
    return get_use_chains(root_getattr_node, terminate)

def _getattr_full_name(getattr):
    return '.'.join([node.attributeNames() for node in getattr])

def _getattr_full_name(getattr):
    return '.'.join(node.s(node.attributeNames()[0]) for node in getattr)

def _get_output_name(node):
    return node.output().debugName()

params = scripted_module.state_dict()
seen = set()
num_count = 0
param_vars = {}

for getattr_node in getattr_nodes:
    if _get_output_name(getattr_node) in seen:
        continue

    getattrs = get_attr_chains(getattr_node)
    for getattr in getattrs:
        seen.update(map(_get_output_name, getattr))
        full_attr = _getattr_full_name(getattr)
        full_attr_node_name = _get_output_name(getattr[-1])
        param_vars[full_attr_node_name] = full_attr
        if full_attr not in params.keys():
            print("fail")
            print(full_attr)
print(param_vars)

nodes = graph.nodes()
ops = []
for node in nodes:
    if node.kind() != "prim::GetAttr":
        print(node)
        ops.append(node)

import LlamaCompiler

def _get_constant(node):
    attribute_names = node.attributeNames()
    if len(attribute_names) == 1:
        attr_name = attribute_names[0]
        ty = node.output().type().kind()
        if ty == "IntType":
            return LlamaCompiler.getConstant(node.i(attr_name), "int", False)
        elif ty == "BoolType":
            return LlamaCompiler.getConstant(bool(node.i(attr_name)), "bool", False)
        elif ty == "FloatType":
            return LlamaCompiler.getConstant(node.f(attr_name), "float", False)
        elif ty == "LongType":
            return LlamaCompiler.getConstant(node.f(attr_name), "long", False)
        elif ty in ["TensorType", "CompleteTensorType"]:
            tensor = node.t(attr_name)
            if tensor.is_cuda:
                tensor = tensor.cpu()
            if len(tensor.shape) == 0:
                print(str(type(tensor.item())))
                return LlamaCompiler.getConstant(tensor.item(), str(type(tensor.item())), False)
            return LlamaCompiler.getConstant(tensor.numpy(), str(type(tensor.item())), True)
        elif ty in ["DeviceObjType", "StringType"]:
            return LlamaCompiler.getConstant(node.s(attr_name), "string", False)
    raise NotImplementedError("Unsupported type: %s" % ty)

def convert_operators(operators, outputs):
    for op in operators:
        if op.kind() == "prim::Constant":
            outputs.append(_get_constant(op))
    
    return outputs

outputs = []
#convert_operators(ops, outputs)
res = []
for node in ops:
    res.append(node.kind())
res = set(res)
for op in res:
    print(op)

print(graph.return_node())
