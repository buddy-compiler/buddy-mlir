mlir_func_define = 'extern "C" void _mlir_ciface_forward(MemRef<float, 3>*, '
with open('params_shape.txt', 'r') as f:
    params_shape = f.readlines()
    for param in params_shape:
        if param.strip() == "":
            continue
        shape_list = param.strip().split(' ')[1].split(',')
        param_str = "MemRef<float, "+str(len(shape_list))+">*, "
        mlir_func_define += param_str
    mlir_func_define += "MemRef<long long, 2>*);"
    print(mlir_func_define)
    for i, param in enumerate(params_shape):
        if param.strip() == "":
            continue
        shape_list = param.strip().split(' ')[1].split(',')
        param_define = "MemRef<float, "+str(len(shape_list))+"> arg"+str(i)+"({"+", ".join(shape_list)+"});"
        print(param_define)
for i in range(355):
    print('ifstream in'+str(i)+'("/home/wlq/buddy-mlir/examples/MLIRLlama/params_data/arg'+str(i)+'.data", ios::in | ios::binary);')
    print("in"+str(i)+".read((char *)(arg"+str(i)+".getData()), sizeof(float) * (arg"+str(i)+".getSize()));")
    print("in"+str(i)+".close();")

call_mlir_function = "_mlir_ciface_forward(&result, "
for i in range(355):
    call_mlir_function += "&arg"+str(i)+", "
call_mlir_function += "&arg355);"
print(call_mlir_function)