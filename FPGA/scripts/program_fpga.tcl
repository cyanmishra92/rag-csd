# Program FPGA script for RAG-CSD

# Open hardware manager
open_hw_manager
connect_hw_server
open_hw_target

# Program device
set_property PROGRAM.FILE {build/rag_csd.runs/impl_1/rag_csd_top.bit} [lindex [get_hw_devices] 0]
program_hw_devices [lindex [get_hw_devices] 0]

# Close hardware manager
close_hw_target
close_hw_manager
