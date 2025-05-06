# Implementation script for RAG-CSD

# Open synthesized design
open_run synth_1

# Run implementation
launch_runs impl_1
wait_on_run impl_1

# Generate reports
open_run impl_1
report_timing_summary -file build/post_route_timing.rpt
report_utilization -file build/post_route_util.rpt
report_power -file build/post_route_power.rpt

# Generate bitstream
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1
