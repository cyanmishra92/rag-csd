//-----------------------------------------------------------------------------
// File: csd_memory_controller.sv
// 
// Description: Memory Controller Module for RAG-CSD
//              Handles access to NAND flash memory in the CSD
//
// Parameters:
//   BUS_WIDTH       - AXI bus width
//   NAND_PAGE_SIZE  - NAND flash page size in bytes
//   NAND_BLOCK_SIZE - NAND flash block size in pages
//   NAND_PLANE_SIZE - NAND flash plane size in blocks
//   NAND_PLANES     - Number of NAND flash planes
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module csd_memory_controller #(
    parameter int BUS_WIDTH = 512,
    parameter int NAND_PAGE_SIZE = 16384,      // 16KB per page
    parameter int NAND_BLOCK_SIZE = 256,       // 256 pages per block
    parameter int NAND_PLANE_SIZE = 2048,      // 2048 blocks per plane
    parameter int NAND_PLANES = 4              // 4 planes (chip dies)
) (
    // Clock and reset
    input  logic                   clk,
    input  logic                   rst_n,
    
    // Memory interface from RAG modules
    input  logic                   mem_rd_en,
    input  logic [31:0]            mem_rd_addr,
    output logic [BUS_WIDTH-1:0]   mem_rd_data,
    output logic                   mem_rd_valid,
    input  logic                   mem_wr_en,
    input  logic [31:0]            mem_wr_addr,
    input  logic [BUS_WIDTH-1:0]   mem_wr_data,
    output logic                   mem_wr_ack,
    
    // NAND flash interface
    output logic                   nand_ce_n,
    output logic                   nand_we_n,
    output logic                   nand_re_n,
    output logic [7:0]             nand_addr,
    inout  logic [7:0]             nand_data,
    input  logic                   nand_rb_n,      // Ready/Busy signal
    
    // Status and control
    output logic [31:0]            status_reg,
    input  logic [31:0]            control_reg
);

    // Memory controller states
    typedef enum logic [3:0] {
        MEM_IDLE,
        MEM_READ_SETUP,
        MEM_READ_CMD,
        MEM_READ_ADDR,
        MEM_READ_DATA,
        MEM_READ_COMPLETE,
        MEM_WRITE_SETUP,
        MEM_WRITE_CMD,
        MEM_WRITE_ADDR,
        MEM_WRITE_DATA,
        MEM_WRITE_COMPLETE,
        MEM_WAIT_READY
    } mem_state_t;
    
    mem_state_t current_state, next_state;
    
    // Cache and buffer management
    localparam int CACHE_SIZE = 1024*1024;  // 1MB cache
    localparam int CACHE_LINES = CACHE_SIZE / NAND_PAGE_SIZE;
    
    // Cache structure
    typedef struct packed {
        logic valid;                  // Cache line valid
        logic dirty;                  // Cache line dirty (needs writeback)
        logic [31:0] addr;            // Physical address
        logic [7:0] data [NAND_PAGE_SIZE-1:0];  // Cached data
    } cache_line_t;
    
    cache_line_t cache [CACHE_LINES-1:0];
    logic [$clog2(CACHE_LINES)-1:0] cache_index;
    logic cache_hit;
    
    // NAND flash controller signals
    logic [31:0] nand_addr_reg;       // NAND address register
    logic [7:0] nand_cmd_reg;         // NAND command register
    logic [7:0] nand_data_out;        // Data output to NAND
    logic [7:0] nand_data_in;         // Data input from NAND
    logic nand_data_dir;              // Data direction (0=input, 1=output)
    logic [31:0] transfer_count;      // Transfer counter
    logic [31:0] transfer_addr;       // Current transfer address
    logic [BUS_WIDTH-1:0] data_buffer; // Data buffer for read/write
    logic [31:0] buffer_offset;       // Offset in buffer
    
    // NAND flash commands
    localparam logic [7:0] NAND_CMD_READ_1 = 8'h00;
    localparam logic [7:0] NAND_CMD_READ_2 = 8'h30;
    localparam logic [7:0] NAND_CMD_PROGRAM_1 = 8'h80;
    localparam logic [7:0] NAND_CMD_PROGRAM_2 = 8'h10;
    localparam logic [7:0] NAND_CMD_ERASE_1 = 8'h60;
    localparam logic [7:0] NAND_CMD_ERASE_2 = 8'hD0;
    localparam logic [7:0] NAND_CMD_RESET = 8'hFF;
    localparam logic [7:0] NAND_CMD_READ_STATUS = 8'h70;
    
    // Tristate buffer for NAND data bus
    assign nand_data = nand_data_dir ? nand_data_out : 8'hZZ;
    assign nand_data_in = nand_data;
    
    // Cache lookup logic
    always_comb begin
        // Calculate physical page address (align to page boundary)
        logic [31:0] page_addr = {mem_rd_addr[31:14], 14'h0};
        cache_index = '0;
        cache_hit = 1'b0;
        
        // Check all cache lines for a hit
        for (int i = 0; i < CACHE_LINES; i++) begin
            if (cache[i].valid && cache[i].addr == page_addr) begin
                cache_index = i;
                cache_hit = 1'b1;
                break;
            end
        end
    end
    
    // State machine for memory controller
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= MEM_IDLE;
            nand_ce_n <= 1'b1;      // Chip disabled
            nand_we_n <= 1'b1;      // Write disabled
            nand_re_n <= 1'b1;      // Read disabled
            nand_addr <= 8'h00;
            nand_data_dir <= 1'b0;  // Input mode
            nand_data_out <= 8'h00;
            nand_addr_reg <= '0;
            nand_cmd_reg <= '0;
            transfer_count <= '0;
            transfer_addr <= '0;
            buffer_offset <= '0;
            mem_rd_valid <= 1'b0;
            mem_wr_ack <= 1'b0;
            status_reg <= 32'h0;
            
            // Initialize cache
            for (int i = 0; i < CACHE_LINES; i++) begin
                cache[i].valid <= 1'b0;
                cache[i].dirty <= 1'b0;
                cache[i].addr <= '0;
            end
        end else begin
            current_state <= next_state;
            
            // Default signal values
            mem_rd_valid <= 1'b0;
            mem_wr_ack <= 1'b0;
            
            case (current_state)
                MEM_IDLE: begin
                    // Handle read request
                    if (mem_rd_en) begin
                        // Check cache first
                        if (cache_hit) begin
                            // Cache hit - return data directly
                            logic [31:0] offset = mem_rd_addr[13:0];  // Offset within page
                            
                            // Assemble data from cache
                            for (int i = 0; i < (BUS_WIDTH/8); i++) begin
                                if (offset + i < NAND_PAGE_SIZE) begin
                                    mem_rd_data[i*8 +: 8] <= cache[cache_index].data[offset + i];
                                end else begin
                                    mem_rd_data[i*8 +: 8] <= 8'h00;  // Padding
                                end
                            end
                            
                            mem_rd_valid <= 1'b1;
                        end else begin
                            // Cache miss - need to fetch from NAND
                            transfer_addr <= {mem_rd_addr[31:14], 14'h0};  // Page aligned address
                            buffer_offset <= mem_rd_addr[13:0];            // Offset within page
                            
                            // Find a cache line to use (simple LRU for now)
                            logic found = 1'b0;
                            for (int i = 0; i < CACHE_LINES; i++) begin
                                if (!cache[i].valid) begin
                                    cache_index <= i;
                                    found = 1'b1;
                                    break;
                                end
                            end
                            
                            if (!found) begin
                                // All cache lines used - pick first non-dirty one
                                // In real implementation, would use more sophisticated replacement
                                cache_index <= 0;
                            end
                        end
                    end
                    
                    // Handle write request
                    if (mem_wr_en) begin
                        // Check if writing to a cached page
                        if (cache_hit) begin
                            // Update cache directly
                            logic [31:0] offset = mem_wr_addr[13:0];  // Offset within page
                            
                            // Update cache data
                            for (int i = 0; i < (BUS_WIDTH/8); i++) begin
                                if (offset + i < NAND_PAGE_SIZE) begin
                                    cache[cache_index].data[offset + i] <= mem_wr_data[i*8 +: 8];
                                end
                            end
                            
                            // Mark cache line as dirty
                            cache[cache_index].dirty <= 1'b1;
                            
                            // Acknowledge write
                            mem_wr_ack <= 1'b1;
                        end else begin
                            // Write miss - need to fetch page first
                            transfer_addr <= {mem_wr_addr[31:14], 14'h0};  // Page aligned address
                            buffer_offset <= mem_wr_addr[13:0];            // Offset within page
                            data_buffer <= mem_wr_data;                    // Store write data
                            
                            // Find a cache line to use (simple LRU for now)
                            logic found = 1'b0;
                            for (int i = 0; i < CACHE_LINES; i++) begin
                                if (!cache[i].valid) begin
                                    cache_index <= i;
                                    found = 1'b1;
                                    break;
                                end
                            end
                            
                            if (!found) begin
                                // All cache lines used - pick first non-dirty one
                                // In real implementation, would use more sophisticated replacement
                                cache_index <= 0;
                            end
                        end
                    end
                end
                
                MEM_READ_SETUP: begin
                    // Setup for NAND read
                    nand_ce_n <= 1'b0;  // Enable chip
                    nand_we_n <= 1'b1;  // Disable write
                    nand_re_n <= 1'b1;  // Disable read
                    nand_cmd_reg <= NAND_CMD_READ_1;
                    
                    // Calculate NAND address components
                    // Assuming simple addressing: row (page) and column (byte in page)
                    // In real hardware, would need to consider plane, block, page addressing
                    nand_addr_reg <= transfer_addr;
                end
                
                MEM_READ_CMD: begin
                    // Send read command
                    nand_data_dir <= 1'b1;  // Output mode
                    nand_data_out <= nand_cmd_reg;
                    nand_we_n <= 1'b0;  // Enable write (to send command)
                    
                    // Latch command on next cycle
                    nand_we_n <= 1'b1;
                end
                
                MEM_READ_ADDR: begin
                    // Send address bytes (5 cycles)
                    nand_data_dir <= 1'b1;  // Output mode
                    
                    // Column address (2 bytes)
                    if (transfer_count < 2) begin
                        nand_data_out <= (transfer_count == 0) ? 
                                         nand_addr_reg[7:0] : nand_addr_reg[15:8];
                    end 
                    // Row address (3 bytes)
                    else begin
                        nand_data_out <= (transfer_count == 2) ? 
                                         nand_addr_reg[23:16] : 
                                         ((transfer_count == 3) ? 
                                          nand_addr_reg[31:24] : 8'h00);
                    end
                    
                    nand_we_n <= 1'b0;  // Enable write (to send address)
                    
                    // Latch address on next cycle
                    nand_we_n <= 1'b1;
                    
                    // Update transfer count
                    transfer_count <= transfer_count + 1;
                    
                    // After last address byte, send second command
                    if (transfer_count == 4) begin
                        nand_cmd_reg <= NAND_CMD_READ_2;
                        transfer_count <= '0;
                    end
                end
                
                MEM_READ_DATA: begin
                    // Wait for ready signal
                    if (!nand_rb_n) begin
                        // NAND is busy
                        nand_ce_n <= 1'b1;  // Disable chip during busy
                    end else begin
                        // NAND is ready - read data
                        nand_ce_n <= 1'b0;  // Enable chip
                        nand_data_dir <= 1'b0;  // Input mode
                        nand_re_n <= 1'b0;  // Enable read
                        
                        // Read data into cache
                        if (transfer_count < NAND_PAGE_SIZE) begin
                            cache[cache_index].data[transfer_count] <= nand_data_in;
                            transfer_count <= transfer_count + 1;
                        end
                        
                        // Latch data on next cycle
                        nand_re_n <= 1'b1;
                    end
                end
                
                MEM_READ_COMPLETE: begin
                    // Read complete - update cache status
                    cache[cache_index].valid <= 1'b1;
                    cache[cache_index].dirty <= 1'b0;
                    cache[cache_index].addr <= transfer_addr;
                    
                    // Disable NAND
                    nand_ce_n <= 1'b1;
                    
                    // Return data to requestor
                    for (int i = 0; i < (BUS_WIDTH/8); i++) begin
                        if (buffer_offset + i < NAND_PAGE_SIZE) begin
                            mem_rd_data[i*8 +: 8] <= cache[cache_index].data[buffer_offset + i];
                        end else begin
                            mem_rd_data[i*8 +: 8] <= 8'h00;  // Padding
                        end
                    end
                    
                    mem_rd_valid <= 1'b1;
                end
                
                MEM_WRITE_SETUP: begin
                    // Setup for NAND write
                    nand_ce_n <= 1'b0;  // Enable chip
                    nand_we_n <= 1'b1;  // Disable write
                    nand_re_n <= 1'b1;  // Disable read
                    nand_cmd_reg <= NAND_CMD_PROGRAM_1;
                    
                    // Calculate NAND address components
                    nand_addr_reg <= transfer_addr;
                    
                    // If we need to read the page first (for write miss)
                    if (!cache[cache_index].valid || 
                        cache[cache_index].addr != transfer_addr) begin
                        // Need to read page first
                        nand_cmd_reg <= NAND_CMD_READ_1;
                    end
                end
                
                MEM_WRITE_CMD: begin
                    // Send write command
                    nand_data_dir <= 1'b1;  // Output mode
                    nand_data_out <= nand_cmd_reg;
                    nand_we_n <= 1'b0;  // Enable write (to send command)
                    
                    // Latch command on next cycle
                    nand_we_n <= 1'b1;
                end
                
                MEM_WRITE_ADDR: begin
                    // Send address bytes (5 cycles)
                    nand_data_dir <= 1'b1;  // Output mode
                    
                    // Column address (2 bytes)
                    if (transfer_count < 2) begin
                        nand_data_out <= (transfer_count == 0) ? 
                                         nand_addr_reg[7:0] : nand_addr_reg[15:8];
                    end 
                    // Row address (3 bytes)
                    else begin
                        nand_data_out <= (transfer_count == 2) ? 
                                         nand_addr_reg[23:16] : 
                                         ((transfer_count == 3) ? 
                                          nand_addr_reg[31:24] : 8'h00);
                    end
                    
                    nand_we_n <= 1'b0;  // Enable write (to send address)
                    
                    // Latch address on next cycle
                    nand_we_n <= 1'b1;
                    
                    // Update transfer count
                    transfer_count <= transfer_count + 1;
                    
                    // After last address byte, move to data phase
                    if (transfer_count == 4) begin
                        transfer_count <= '0;
                    end
                end
                
                MEM_WRITE_DATA: begin
                    // Write data to NAND
                    nand_data_dir <= 1'b1;  // Output mode
                    
                    if (transfer_count < NAND_PAGE_SIZE) begin
                        nand_data_out <= cache[cache_index].data[transfer_count];
                        nand_we_n <= 1'b0;  // Enable write
                        
                        // Latch data on next cycle
                        nand_we_n <= 1'b1;
                        
                        transfer_count <= transfer_count + 1;
                    end
                    
                    // After last data byte, send program command
                    if (transfer_count == NAND_PAGE_SIZE - 1) begin
                        nand_cmd_reg <= NAND_CMD_PROGRAM_2;
                    end
                end
                
                MEM_WRITE_COMPLETE: begin
                    // Send final program command
                    nand_data_dir <= 1'b1;  // Output mode
                    nand_data_out <= nand_cmd_reg;
                    nand_we_n <= 1'b0;  // Enable write
                    
                    // Latch command on next cycle
                    nand_we_n <= 1'b1;
                    
                    // Mark cache line as clean
                    cache[cache_index].dirty <= 1'b0;
                    
                    // Acknowledge write to requestor
                    mem_wr_ack <= 1'b1;
                end
                
                MEM_WAIT_READY: begin
                    // Wait for NAND to be ready
                    if (nand_rb_n) begin
                        // NAND is ready - operation complete
                        nand_ce_n <= 1'b1;  // Disable chip
                    end
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            MEM_IDLE: begin
                if (mem_rd_en && !cache_hit) begin
                    next_state = MEM_READ_SETUP;
                end else if (mem_wr_en && !cache_hit) begin
                    next_state = MEM_WRITE_SETUP;
                end
            end
            
            MEM_READ_SETUP: begin
                next_state = MEM_READ_CMD;
            end
            
            MEM_READ_CMD: begin
                next_state = MEM_READ_ADDR;
            end
            
            MEM_READ_ADDR: begin
                if (transfer_count >= 4 && nand_cmd_reg == NAND_CMD_READ_1) begin
                    next_state = MEM_READ_CMD;
                end else if (transfer_count >= 4 && nand_cmd_reg == NAND_CMD_READ_2) begin
                    next_state = MEM_WAIT_READY;
                    transfer_count = '0;
                end
            end
            
            MEM_WAIT_READY: begin
                if (nand_rb_n) begin
                    if (nand_cmd_reg == NAND_CMD_READ_2) begin
                        next_state = MEM_READ_DATA;
                    end else if (nand_cmd_reg == NAND_CMD_PROGRAM_2) begin
                        next_state = MEM_IDLE;
                    end
                end
            end
            
            MEM_READ_DATA: begin
                if (transfer_count >= NAND_PAGE_SIZE) begin
                    next_state = MEM_READ_COMPLETE;
                end
            end
            
            MEM_READ_COMPLETE: begin
                next_state = MEM_IDLE;
            end
            
            MEM_WRITE_SETUP: begin
                next_state = MEM_WRITE_CMD;
            end
            
            MEM_WRITE_CMD: begin
                next_state = MEM_WRITE_ADDR;
            end
            
            MEM_WRITE_ADDR: begin
                if (transfer_count >= 4) begin
                    next_state = MEM_WRITE_DATA;
                    transfer_count = '0;
                end
            end
            
            MEM_WRITE_DATA: begin
                if (transfer_count >= NAND_PAGE_SIZE) begin
                    next_state = MEM_WRITE_COMPLETE;
                end
            end
            
            MEM_WRITE_COMPLETE: begin
                next_state = MEM_WAIT_READY;
            end
            
            default: next_state = MEM_IDLE;
        endcase
    end

endmodule
