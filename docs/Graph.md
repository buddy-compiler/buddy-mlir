# Graph Computation Framework
## Overview
This design document describes a graph computation framework used in the Buddy Compiler frontend. The main purpose of this framework is to build, optimize, and ultimately generate MLIR (Multi-Level Intermediate Representation) modules that can run on various hardware devices. This document will detail the design, core classes, and their functionalities and implementation details.
## Core Classes and Modules
### 'Graph'Class
The Graph class is the core of the framework, representing a graph-level expression. It serves as the structure of a model compute graph, with the main goal of converting the graph into an equivalent MLIR module. 
Key Attributes:
- `_body`: List[Op] - Stores the sequence of operation nodes in the graph.
- `_inputs`: List[TensorMeta] - Represents the model inputs as TensorMeta objects.
- `_fake_params`: List[TensorMeta] - Represents fake parameters as TensorMeta objects.
- `device`: str - Specifies the hardware device for graph runtime, defaulting to "cpu".
- `_imported_module`: Union[None, ImportedModuleType] - The imported MLIR module after compilation.
- `_ops_registry`: dict - Stores the operation lowering strategies for the graph.
- `_func_name`: str - The function name for the MLIR module.
- `_ctx`: ir.Context - The context of the MLIR module.
- `_output_memref`: Union[None, ctypes.POINTER] - The memref pointer in the MLIR function output.
- `_output_descriptor`: Union[None, OutputDescriptorType] - The output descriptor for the MLIR function.
- `execution_engine`: Union[None, ExecutionEngineType] - The execution engine for the graph.

Key Functions:
- `__init__`: Initializes the Graph object with inputs, fake parameters, operation registry, and function name.
- `add_node`: Adds an operation node to the graph's body.
- `init_op_group`: Initializes operation groups within the graph based on the operations provided.
- `fuse_ops`: Fuses operations within the graph according to provided fusion patterns.
- `perform`: Applies a series of transformation functions to the graph.
- `lower_to_top_level_ir`: Lowers the graph to top-level MLIR dialects.
- `lower_to_llvm_ir`: Further lowers the graph to LLVM IR.
- `compile`: Compiles the graph, progressing from Buddy Graph to LLVM IR.

### 'GraphImporter'Class
The GraphImporter class is responsible for importing the Buddy graph and generating a high-level MLIR module. The primary duty of this class is to map the graph's operation nodes to MLIR dialects and organize them into an MLIR module.
Key Attributes:
- `_symbol_table`: dict - A dictionary to keep track of symbols.
- `_body`: List[Op] - The FX graph module to be imported.
- `_func_name`: str - Name of the generated MLIR function.
- `_inputs`: List[TensorMeta] - Input tensors of the FX graph.
- `_num_input_visited`: int - The number of input nodes that have been visited.
- `_module`: mlir.ir.Module - The generated MLIR module.
- `_ops_registry`: dict - Registry for the candidate operations.

Key Functions:
- `__init__`: Initializes the GraphImporter object, preparing to import the Buddy graph.
- `import_graph`: Imports the Buddy graph and generates a high-level MLIR module.
- `import_main_graph`: Imports the Buddy main graph, organizes all subgraphs, and generates a high-level MLIR module.
- `_import_placeholder`: Imports a placeholder node from the Buddy graph.
- `_import_op`: Imports an operation node from the Buddy graph.
- `get_output_nodes`: Retrieves output nodes from the lowered MLIR function.
### 'GraphDriver'Class
The GraphDriver class is responsible for managing the execution of a computational graph. It handles the division of the graph into subgraphs and orchestrates the execution order by constructing a main graph that calls these subgraphs in the correct sequence.
Key Attributes:
- `_graph`: Graph - The computational graph associated with this driver.
- `_subgraphs`: Dict[str, Graph] - A dictionary mapping subgraph names to their corresponding subgraphs.
- `_subgraphs_inputs`: Dict[str, List[str]] - A dictionary mapping subgraph names to their input placeholders.
- `_subgraphs_outputs`: Dict[str, List[str]] - A dictionary mapping subgraph names to their output operation results.

Key Functions:
- `__init__`: Initializes the GraphDriver object with a given computational graph, dividing it into subgraphs.
- `build_subgraph_by_group`: Builds subgraphs from the main graph by grouping operations and identifying their inputs and outputs.
- `construct_main_graph`: Constructs the main graph that coordinates the execution of subgraphs, potentially performing parameter packing.
## Graph Construction and Subdivision
### Building and Grouping Operations in the Graph
The Graph class allows for the construction of a computational graph by adding operation nodes, which represent various computations or data transformations. Once the graph is built, operations can be grouped into subgraphs using the init_op_group method, which organizes operations into groups that can later be optimized and executed independently.

### Subdividing the Graph with `GraphDriver`
The `GraphDriver` class takes a complete `Graph` and subdivides it into smaller subgraphs. This subdivision is crucial for managing complex computations, as it allows for modular execution and easier optimization. The `build_subgraph_by_group` method identifies inputs and outputs for each subgraph, ensuring that data dependencies are correctly managed across subgraphs.

## Graph Optimization and Execution
### Operation Fusion and Graph Optimization
After grouping the operations into subgraphs, the `fuse_ops` method in the Graph class can be used to fuse operations based on provided patterns. This step optimizes the graph by reducing the number of operations or rearranging them for better performance on the target hardware.

The `perform` method allows applying a series of transformations to the graph, further refining its structure and preparing it for execution.

### Constructing and Executing the Main Graph
The `GraphDriver` class's `construct_main_graph` method creates the main computational graph, which orchestrates the execution of the subgraphs. This method can incorporate optimizations such as parameter packing, depending on the needs of the execution environment.

Once the main graph is constructed, the entire computational structure can be compiled using the `compile` method in the `Graph` class, which lowers the graph to MLIR and LLVM IR, making it ready for execution on the target hardware.

## Graph Lowering and Compilation
### Lowering to Top-Level MLIR Dialects
The `lower_to_top_level_ir` method in the `Graph` class lowers the graph to top-level MLIR dialects. This involves converting the graph's operations into equivalent MLIR representations, creating a module that can be optimized further or converted to LLVM IR.

### Lowering to LLVM IR and Compilation
Following the MLIR lowering, the `lower_to_llvm_ir` method further reduces the graph to LLVM IR, applying various optimizations specific to the LLVM framework. This final step in the compilation process makes the graph executable on a wide range of hardware, completing the transformation from high-level graph representation to low-level executable code.


