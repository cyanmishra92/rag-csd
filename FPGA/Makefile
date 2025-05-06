# Makefile for RAG-CSD FPGA implementation

# Xilinx Vivado settings
VIVADO_BIN = vivado
VIVADO_MODE = -mode batch

# Project settings
PROJECT_NAME = rag_csd
PART = xcvu37p-fsvh2892-2L-e  # Example Xilinx Ultrascale+ part for CSD

# Directories
RTL_DIR = rtl
SIM_DIR = sim
TB_DIR = testbench
CONSTR_DIR = constraints
SCRIPTS_DIR = scripts
BUILD_DIR = build

# Targets
.PHONY: all clean sim synth impl program

all: synth impl

# Simulation
sim:
	@echo "Running simulation..."
	xvlog --sv $(SIM_DIR)/sim_top.sv $(SIM_DIR)/mem_model.sv $(TB_DIR)/*.sv $(RTL_DIR)/*/*.sv
	xelab -debug typical -top rag_csd_tb -snapshot rag_csd_tb_snapshot
	xsim rag_csd_tb_snapshot -R

# Synthesis
synth:
	@echo "Running synthesis..."
	mkdir -p $(BUILD_DIR)
	$(VIVADO_BIN) $(VIVADO_MODE) -source $(SCRIPTS_DIR)/synth.tcl

# Implementation
impl:
	@echo "Running implementation..."
	$(VIVADO_BIN) $(VIVADO_MODE) -source $(SCRIPTS_DIR)/implement.tcl

# Program FPGA
program:
	@echo "Programming FPGA..."
	$(VIVADO_BIN) $(VIVADO_MODE) -source $(SCRIPTS_DIR)/program_fpga.tcl

# Clean
clean:
	rm -rf $(BUILD_DIR)
	rm -rf *.log *.jou *.str
	rm -rf .Xil
	rm -rf xsim.dir
	rm -rf *.pb
