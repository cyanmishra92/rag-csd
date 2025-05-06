# Synthesis script for RAG-CSD

# Create project
create_project rag_csd build -part xcvu37p-fsvh2892-2L-e -force

# Add source files
add_files rtl/core
add_files rtl/encoder
add_files rtl/retrieval
add_files rtl/augmentation
add_files rtl/memory
add_files rtl/interface

# Add constraints
add_files -fileset constrs_1 constraints/timing_constraints.xdc
add_files -fileset constrs_1 constraints/pin_assignments.xdc

# Set top module
set_property top rag_csd_top [current_fileset]

# Run synthesis
launch_runs synth_1
wait_on_run synth_1

# Generate reports
open_run synth_1
report_timing_summary -file build/post_synth_timing.rpt
report_utilization -file build/post_synth_util.rpt
