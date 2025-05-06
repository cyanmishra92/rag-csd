//-----------------------------------------------------------------------------
// File: csd_interface_controller.sv
// 
// Description: CSD Interface Controller Module for RAG-CSD
//              Handles communication between host system and RAG modules
//              via PCIe/NVMe interface
//
// Parameters:
//   BUS_WIDTH       - AXI bus width
//   MAX_PAYLOAD     - Maximum payload size
//   MAX_CMD_QUEUE   - Maximum command queue size
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module csd_interface_controller #(
    parameter int BUS_WIDTH = 512,
    parameter int MAX_PAYLOAD = 4096,    // Maximum payload size in bytes
    parameter int MAX_CMD_QUEUE = 16     // Maximum command queue size
) (
    // PCIe/CSD interface clock and reset
    input  logic                   pcie_clk,
    input  logic                   pcie_rst_n,
    
    // PCIe/CSD interface (simplified for implementation)
    input  logic [BUS_WIDTH-1:0]   pcie_rx_data,
    input  logic                   pcie_rx_valid,
    output logic                   pcie_rx_ready,
    output logic [BUS_WIDTH-1:0]   pcie_tx_data,
    output logic                   pcie_tx_valid,
    input  logic                   pcie_tx_ready,
    
    // Internal RAG module interface clock and reset
    output logic                   rag_clk,
    output logic                   rag_rst_n,
    
    // RAG module data interface
    output logic [BUS_WIDTH-1:0]   rag_query_data,
    output logic                   rag_query_valid,
    input  logic                   rag_query_ready,
    input  logic [BUS_WIDTH-1:0]   rag_result_data,
    input  logic                   rag_result_valid,
    output logic                   rag_result_ready,
    
    // AXI-Lite control interface
    output logic                   s_axil_awvalid,
    input  logic                   s_axil_awready,
    output logic [31:0]            s_axil_awaddr,
    output logic                   s_axil_wvalid,
    input  logic                   s_axil_wready,
    output logic [31:0]            s_axil_wdata,
    input  logic                   s_axil_bvalid,
    output logic                   s_axil_bready,
    input  logic [1:0]             s_axil_bresp,
    output logic                   s_axil_arvalid,
    input  logic                   s_axil_arready,
    output logic [31:0]            s_axil_araddr,
    input  logic                   s_axil_rvalid,
    output logic                   s_axil_rready,
    input  logic [31:0]            s_axil_rdata,
    input  logic [1:0]             s_axil_rresp
);

    // NVMe command types
    typedef enum logic [7:0] {
        CMD_RAG_QUERY = 8'h80,     // Custom RAG query command
        CMD_GET_RESULTS = 8'h81,   // Get query results
        CMD_ABORT = 8'h82,         // Abort current operation
        CMD_CONFIG = 8'h83         // Configure parameters
    } nvme_cmd_t;

    // Interface states
    typedef enum logic [3:0] {
        IF_IDLE,
        IF_CMD_FETCH,
        IF_CMD_DECODE,
        IF_DATA_FETCH,
        IF_PROCESS_RAG,
        IF_WAIT_COMPLETION,
        IF_SEND_RESPONSE,
        IF_SEND_DATA
    } interface_state_t;
    
    interface_state_t current_state, next_state;
    
    // Command queue
    typedef struct packed {
        logic valid;                // Command is valid
        logic [7:0] cmd_type;       // Command type
        logic [7:0] cmd_id;         // Command ID
        logic [15:0] cmd_flags;     // Command flags
        logic [31:0] cmd_params[4]; // Command parameters
        logic [31:0] data_addr;     // Data buffer address
        logic [31:0] data_length;   // Data buffer length
    } cmd_entry_t;
    
    cmd_entry_t cmd_queue [MAX_CMD_QUEUE-1:0];
    logic [$clog2(MAX_CMD_QUEUE)-1:0] cmd_wr_ptr, cmd_rd_ptr;
    logic cmd_queue_empty, cmd_queue_full;
    
    // Data buffers
    logic [7:0] data_buffer [MAX_PAYLOAD-1:0];
    logic [31:0] data_length;
    logic [31:0] data_offset;
    
    // Status and command tracking
    logic [7:0] current_cmd_id;
    logic [31:0] status_reg;
    
    // Clock domain crossing logic
    logic pcie_to_rag_valid, rag_to_pcie_valid;
    logic pcie_to_rag_ready, rag_to_pcie_ready;
    
    // Cross-domain synchronization for reset
    logic [2:0] rag_rst_sync;
    
    // Generate RAG clock (in real hardware would be from PLL)
    // For simulation, use PCIe clock
    assign rag_clk = pcie_clk;
    
    // Reset synchronizer
    always_ff @(posedge rag_clk or negedge pcie_rst_n) begin
        if (!pcie_rst_n) begin
            rag_rst_sync <= 3'b0;
            rag_rst_n <= 1'b0;
        end else begin
            rag_rst_sync <= {rag_rst_sync[1:0], 1'b1};
            rag_rst_n <= rag_rst_sync[2];
        end
    end
    
    // Command queue management
    assign cmd_queue_empty = (cmd_rd_ptr == cmd_wr_ptr) && !cmd_queue[cmd_rd_ptr].valid;
    assign cmd_queue_full = (cmd_rd_ptr == cmd_wr_ptr) && cmd_queue[cmd_rd_ptr].valid;
    
    // State machine for CSD interface controller
    always_ff @(posedge pcie_clk or negedge pcie_rst_n) begin
        if (!pcie_rst_n) begin
            current_state <= IF_IDLE;
            cmd_wr_ptr <= '0;
            cmd_rd_ptr <= '0;
            data_length <= '0;
            data_offset <= '0;
            current_cmd_id <= '0;
            status_reg <= 32'h0;
            
            // Reset PCIe interface signals
            pcie_rx_ready <= 1'b0;
            pcie_tx_valid <= 1'b0;
            pcie_tx_data <= '0;
            
            // Reset RAG interface signals
            rag_query_valid <= 1'b0;
            rag_query_data <= '0;
            rag_result_ready <= 1'b0;
            
            // Reset AXI-Lite signals
            s_axil_awvalid <= 1'b0;
            s_axil_awaddr <= '0;
            s_axil_wvalid <= 1'b0;
            s_axil_wdata <= '0;
            s_axil_bready <= 1'b0;
            s_axil_arvalid <= 1'b0;
            s_axil_araddr <= '0;
            s_axil_rready <= 1'b0;
            
            // Initialize command queue
            for (int i = 0; i < MAX_CMD_QUEUE; i++) begin
                cmd_queue[i].valid <= 1'b0;
            end
        end else begin
            current_state <= next_state;
            
            // Default values for interface signals
            pcie_tx_valid <= 1'b0;
            
            case (current_state)
                IF_IDLE: begin
                    // Accept new commands if queue not full
                    pcie_rx_ready <= !cmd_queue_full;
                    
                    // Process incoming command
                    if (pcie_rx_valid && pcie_rx_ready) begin
                        // Identify command header
                        if (pcie_rx_data[7:0] >= 8'h80) begin  // Custom commands start at 0x80
                            // Store command in queue
                            cmd_queue[cmd_wr_ptr].valid <= 1'b1;
                            cmd_queue[cmd_wr_ptr].cmd_type <= pcie_rx_data[7:0];
                            cmd_queue[cmd_wr_ptr].cmd_id <= pcie_rx_data[15:8];
                            cmd_queue[cmd_wr_ptr].cmd_flags <= pcie_rx_data[31:16];
                            cmd_queue[cmd_wr_ptr].cmd_params[0] <= pcie_rx_data[63:32];
                            cmd_queue[cmd_wr_ptr].cmd_params[1] <= pcie_rx_data[95:64];
                            cmd_queue[cmd_wr_ptr].cmd_params[2] <= pcie_rx_data[127:96];
                            cmd_queue[cmd_wr_ptr].cmd_params[3] <= pcie_rx_data[159:128];
                            cmd_queue[cmd_wr_ptr].data_addr <= pcie_rx_data[191:160];
                            cmd_queue[cmd_wr_ptr].data_length <= pcie_rx_data[223:192];
                            
                            // Update write pointer
                            cmd_wr_ptr <= (cmd_wr_ptr + 1) % MAX_CMD_QUEUE;
                        end
                    end
                end
                
                IF_CMD_FETCH: begin
                    // Fetch command from queue
                    if (!cmd_queue_empty) begin
                        current_cmd_id <= cmd_queue[cmd_rd_ptr].cmd_id;
                        
                        // For commands with data payload, prepare for data fetch
                        if (cmd_queue[cmd_rd_ptr].data_length > 0 && 
                            cmd_queue[cmd_rd_ptr].cmd_type == CMD_RAG_QUERY) begin
                            data_length <= cmd_queue[cmd_rd_ptr].data_length;
                            data_offset <= '0;
                        end
                    end
                end
                
                IF_CMD_DECODE: begin
                    // Decode and execute command
                    case (cmd_queue[cmd_rd_ptr].cmd_type)
                        CMD_RAG_QUERY: begin
                            // Handle RAG query - needs to fetch data first
                            if (cmd_queue[cmd_rd_ptr].data_length > 0) begin
                                // Set up to fetch data from host
                                pcie_rx_ready <= 1'b1;
                            end
                        end
                        
                        CMD_GET_RESULTS: begin
                            // Get RAG results - check if ready
                            if (status_reg[0]) begin  // Results ready bit
                                // Will send data in IF_SEND_DATA state
                            end else begin
                                // Results not ready yet
                                // Send immediate completion with error status
                                pcie_tx_valid <= 1'b1;
                                pcie_tx_data <= {448'h0,   // Reserved
                                                32'h0001,  // Status: Not ready
                                                8'h00,     // Reserved
                                                cmd_queue[cmd_rd_ptr].cmd_id,  // Command ID
                                                8'h80};    // Response code
                                
                                // Mark command as completed
                                cmd_queue[cmd_rd_ptr].valid <= 1'b0;
                                cmd_rd_ptr <= (cmd_rd_ptr + 1) % MAX_CMD_QUEUE;
                            end
                        end
                        
                        CMD_ABORT: begin
                            // Abort current operation
                            rag_query_valid <= 1'b0;
                            status_reg <= 32'h0;  // Clear status
                            
                            // Send immediate completion
                            pcie_tx_valid <= 1'b1;
                            pcie_tx_data <= {448'h0,   // Reserved
                                            32'h0,     // Status: Success
                                            8'h00,     // Reserved
                                            cmd_queue[cmd_rd_ptr].cmd_id,  // Command ID
                                            8'h81};    // Response code
                            
                            // Mark command as completed
                            cmd_queue[cmd_rd_ptr].valid <= 1'b0;
                            cmd_rd_ptr <= (cmd_rd_ptr + 1) % MAX_CMD_QUEUE;
                        end
                        
                        CMD_CONFIG: begin
                            // Configure RAG parameters through AXI-Lite
                            if (!s_axil_awvalid && !s_axil_wvalid) begin
                                // Start AXI-Lite write transaction
                                s_axil_awvalid <= 1'b1;
                                s_axil_awaddr <= cmd_queue[cmd_rd_ptr].cmd_params[0];
                                s_axil_wvalid <= 1'b1;
                                s_axil_wdata <= cmd_queue[cmd_rd_ptr].cmd_params[1];
                                s_axil_bready <= 1'b1;
                            end else if (s_axil_bvalid && s_axil_bready) begin
                                // Write transaction complete
                                s_axil_awvalid <= 1'b0;
                                s_axil_wvalid <= 1'b0;
                                s_axil_bready <= 1'b0;
                                
                                // Send completion
                                pcie_tx_valid <= 1'b1;
                                pcie_tx_data <= {448'h0,   // Reserved
                                                32'h0,     // Status: Success
                                                8'h00,     // Reserved
                                                cmd_queue[cmd_rd_ptr].cmd_id,  // Command ID
                                                8'h82};    // Response code
                                
                                // Mark command as completed
                                cmd_queue[cmd_rd_ptr].valid <= 1'b0;
                                cmd_rd_ptr <= (cmd_rd_ptr + 1) % MAX_CMD_QUEUE;
                            end
                        end
                        
                        default: begin
                            // Unknown command - send error response
                            pcie_tx_valid <= 1'b1;
                            pcie_tx_data <= {448'h0,   // Reserved
                                            32'h0002,  // Status: Invalid command
                                            8'h00,     // Reserved
                                            cmd_queue[cmd_rd_ptr].cmd_id,  // Command ID
                                            8'hFF};    // Response code
                            
                            // Mark command as completed
                            cmd_queue[cmd_rd_ptr].valid <= 1'b0;
                            cmd_rd_ptr <= (cmd_rd_ptr + 1) % MAX_CMD_QUEUE;
                        end
                    endcase
                end
                
                IF_DATA_FETCH: begin
                    // Fetch data payload for RAG query
                    if (pcie_rx_valid && pcie_rx_ready) begin
                        // Store received data in buffer
                        for (int i = 0; i < (BUS_WIDTH/8); i++) begin
                            if (data_offset + i < data_length && data_offset + i < MAX_PAYLOAD) begin
                                data_buffer[data_offset + i] <= pcie_rx_data[i*8 +: 8];
                            end
                        end
                        
                        data_offset <= data_offset + (BUS_WIDTH/8);
                        
                        // Check if all data received
                        if (data_offset + (BUS_WIDTH/8) >= data_length) begin
                            pcie_rx_ready <= 1'b0;
                        end
                    end
                end
                
                IF_PROCESS_RAG: begin
                    // Send query data to RAG module
                    if (!rag_query_valid || (rag_query_valid && rag_query_ready)) begin
                        if (data_offset < data_length) begin
                            rag_query_valid <= 1'b1;
                            
                            // Prepare data for RAG module
                            for (int i = 0; i < (BUS_WIDTH/8); i++) begin
                                if (data_offset + i < data_length) begin
                                    rag_query_data[i*8 +: 8] <= data_buffer[data_offset + i];
                                end else begin
                                    rag_query_data[i*8 +: 8] <= 8'h00;  // Padding
                                end
                            end
                            
                            // Last data word
                            if (data_offset + (BUS_WIDTH/8) >= data_length) begin
                                // Set last flag in control register
                                status_reg[1] <= 1'b1;  // Query in progress bit
                            end
                            
                            data_offset <= data_offset + (BUS_WIDTH/8);
                        end else begin
                            rag_query_valid <= 1'b0;
                        end
                    end
                    
                    // Send completion to host to acknowledge query received
                    if (data_offset >= data_length && 
                        !rag_query_valid && 
                        !pcie_tx_valid) begin
                        
                        pcie_tx_valid <= 1'b1;
                        pcie_tx_data <= {448'h0,   // Reserved
                                        32'h0,     // Status: Success
                                        8'h00,     // Reserved
                                        cmd_queue[cmd_rd_ptr].cmd_id,  // Command ID
                                        8'h83};    // Response code
                        
                        // Mark command as completed
                        cmd_queue[cmd_rd_ptr].valid <= 1'b0;
                        cmd_rd_ptr <= (cmd_rd_ptr + 1) % MAX_CMD_QUEUE;
                    end
                end
                
                IF_WAIT_COMPLETION: begin
                    // Enable result ready signal
                    rag_result_ready <= 1'b1;
                    
                    // Buffer results data as it arrives
                    if (rag_result_valid && rag_result_ready) begin
                        // Store result data
                        for (int i = 0; i < (BUS_WIDTH/8); i++) begin
                            if (data_offset + i < MAX_PAYLOAD) begin
                                data_buffer[data_offset + i] <= rag_result_data[i*8 +: 8];
                            end
                        end
                        
                        data_offset <= data_offset + (BUS_WIDTH/8);
                        data_length <= data_offset + (BUS_WIDTH/8);
                        
                        // Check for last flag
                        if (rag_result_data[0]) begin
                            // Query complete, set results ready flag
                            status_reg[0] <= 1'b1;  // Results ready bit
                            status_reg[1] <= 1'b0;  // Clear query in progress bit
                        end
                    end
                end
                
                IF_SEND_RESPONSE: begin
                    // Send response header
                    pcie_tx_valid <= 1'b1;
                    pcie_tx_data <= {448'h0,               // Reserved
                                    status_reg,            // Status register
                                    8'h00,                 // Reserved
                                    current_cmd_id,        // Command ID
                                    8'h84};                // Response code
                end
                
                IF_SEND_DATA: begin
                    // Send result data to host
                    pcie_tx_valid <= 1'b1;
                    
                    // Prepare data packet
                    for (int i = 0; i < (BUS_WIDTH/8); i++) begin
                        if (data_offset + i < data_length) begin
                            pcie_tx_data[i*8 +: 8] <= data_buffer[data_offset + i];
                        end else begin
                            pcie_tx_data[i*8 +: 8] <= 8'h00;  // Padding
                        end
                    end
                    
                    // Update data offset if transfer successful
                    if (pcie_tx_ready) begin
                        data_offset <= data_offset + (BUS_WIDTH/8);
                        
                        // Check if all data sent
                        if (data_offset + (BUS_WIDTH/8) >= data_length) begin
                            pcie_tx_valid <= 1'b0;
                            
                            // Mark command as completed
                            cmd_queue[cmd_rd_ptr].valid <= 1'b0;
                            cmd_rd_ptr <= (cmd_rd_ptr + 1) % MAX_CMD_QUEUE;
                        end
                    end
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            IF_IDLE: begin
                if (!cmd_queue_empty) begin
                    next_state = IF_CMD_FETCH;
                end
            end
            
            IF_CMD_FETCH: begin
                if (!cmd_queue_empty) begin
                    next_state = IF_CMD_DECODE;
                end
            end
            
            IF_CMD_DECODE: begin
                if (!cmd_queue_empty) begin
                    case (cmd_queue[cmd_rd_ptr].cmd_type)
                        CMD_RAG_QUERY: begin
                            if (cmd_queue[cmd_rd_ptr].data_length > 0) begin
                                next_state = IF_DATA_FETCH;
                            end else begin
                                next_state = IF_PROCESS_RAG;
                            end
                        end
                        
                        CMD_GET_RESULTS: begin
                            if (status_reg[0]) begin  // Results ready bit
                                next_state = IF_SEND_RESPONSE;
                            end
                        end
                        
                        default: begin
                            next_state = IF_IDLE;
                        end
                    endcase
                end
            end
            
            IF_DATA_FETCH: begin
                if (data_offset >= data_length) begin
                    next_state = IF_PROCESS_RAG;
                end
            end
            
            IF_PROCESS_RAG: begin
                if (data_offset >= data_length && !rag_query_valid) begin
                    next_state = IF_WAIT_COMPLETION;
                end
            end
            
            IF_WAIT_COMPLETION: begin
                if (status_reg[0]) begin  // Results ready bit
                    next_state = IF_IDLE;
                end
            end
            
            IF_SEND_RESPONSE: begin
                if (pcie_tx_ready && pcie_tx_valid) begin
                    next_state = IF_SEND_DATA;
                    data_offset = '0;  // Reset data offset for sending
                end
            end
            
            IF_SEND_DATA: begin
                if (data_offset >= data_length) begin
                    next_state = IF_IDLE;
                end
            end
            
            default: next_state = IF_IDLE;
        endcase
    end

endmodule
