# RUN: %PYTHON %s 2>&1 | FileCheck %s
import torch

from buddy.compiler.frontend import DynamoCompiler


# Define the target function or model.
def foo(x, y):
    return x * y + x


# Define the input data.
float32_in1 = torch.randn(10).to(torch.float32)
float32_in2 = torch.randn(10).to(torch.float32)

# Test the default dynamo compiler importer mode.
dynamo_compiler_default = DynamoCompiler()
graphs = dynamo_compiler_default.importer(foo, *(float32_in1, float32_in2))

# Ensure no output is printed in the default mode.
# CHECK-NOT: .

# Test the dynamo compiler verbose mode.
dynamo_compiler_verbose_on = DynamoCompiler(verbose=True)
graphs = dynamo_compiler_verbose_on.importer(foo, *(float32_in1, float32_in2))

# Test output in the verbose mode.
# CHECK: placeholder
# CHECK: placeholder
# CHECK: call_function
# CHECK: call_function
# CHECK: output

# Test the dynamo compiler verbose mode off.
dynamo_compiler_verbose_off = DynamoCompiler(verbose=False)
graphs = dynamo_compiler_verbose_off.importer(foo, *(float32_in1, float32_in2))

# Ensure no output is printed when the verbose mode is off.
# CHECK-NOT: .
