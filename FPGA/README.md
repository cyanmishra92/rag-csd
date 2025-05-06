# RAG-CSD FPGA Implementation

This directory contains the FPGA implementation for the Retrieval-Augmented Generation (RAG) system on Xilinx/Samsung Computational Storage Devices (CSDs).

## Directory Structure

- `rtl/`: Register-Transfer Level design files
  - `core/`: Top-level and core modules
  - `encoder/`: Transformer encoder modules
  - `retrieval/`: Vector similarity and retrieval modules
  - `augmentation/`: Query augmentation modules
  - `memory/`: Memory interface modules
  - `interface/`: External interface modules
- `constraints/`: Timing and pin constraints
- `sim/`: Simulation files
- `testbench/`: Testbenches for verification
- `scripts/`: Scripts for synthesis and implementation
- `doc/`: Documentation

## Build Instructions

1. Set up Xilinx Vivado environment
2. Run `make` to synthesize and implement the design
3. Run `make sim` to run simulation
4. Run `make program` to program the FPGA

## Implementation Details

The implementation targets Xilinx UltraScale+ FPGAs typically used in Samsung CSDs. The design includes:

1. **Encoder Module**: Implements lightweight transformer for embedding generation
2. **Retrieval Module**: Implements efficient vector similarity search
3. **Augmentation Module**: Combines query with retrieved documents
4. **Interface Controllers**: Connect to CSD NVMe/PCIe interface
5. **Memory Controllers**: Manage access to NAND flash storage

## Performance

The implementation is optimized for:
- High throughput vector similarity search
- Efficient memory access patterns
- Parallel processing of vector operations

## Integration

See `doc/integration_guide.md` for details on integrating with the RAG-CSD software stack.
