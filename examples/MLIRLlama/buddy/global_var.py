
global _global_dict
_global_dict = {"param-to-mlir":False, 
                "params-write-path":"/buddy-mlir/examples/MLIRLlama",
                "params-pack":True}

def global_var_set_value(key, value):
    _global_dict[key] = value

def global_var_get_value(key):
    return _global_dict[key]
    