//-----------------------------------------------------------------------------
// File: axi_lite_interface.sv
// 
// Description: AXI-Lite Interface Module for RAG-CSD
//              Handles configuration and control registers access
//
// Parameters:
//   ADDR_WIDTH     - Address width
//   DATA_WIDTH     - Data width (32 bits for AXI-Lite)
//   NUM_REGISTERS  - Number of control/status registers
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module axi_lite_interface #(
    parameter int ADDR_WIDTH = 32,
    parameter int DATA_WIDTH = 32,
    parameter int NUM_REGISTERS = 16
) (
    // Clock and reset
    input  logic                   clk,
    input  logic                   rst_n,
    
    // AXI-Lite slave interface
    input  logic                   s_axil_awvalid,
    output logic                   s_axil_awready,
    input  logic [ADDR_WIDTH-1:0]  s_axil_awaddr,
    input  logic                   s_axil_wvalid,
    output logic                   s_axil_wready,
    input  logic [DATA_WIDTH-1:0]  s_axil_wdata,
    output logic                   s_axil_bvalid,
    input  logic                   s_axil_bready,
    output logic [1:0]             s_axil_bresp,
    input  logic                   s_axil_arvalid,
    output logic                   s_axil_arready,
    input  logic [ADDR_WIDTH-1:0]  s_axil_araddr,
    output logic                   s_axil_rvalid,
    input  logic                   s_axil_rready,
    output logic [DATA_WIDTH-1:0]  s_axil_rdata,
    output logic [1:0]             s_axil_rresp,
    
    // Register interface
    output logic [NUM_REGISTERS-1:0][DATA_WIDTH-1:0] registers_out,
    input  logic [NUM_REGISTERS-1:0][DATA_WIDTH-1:0] registers_in,
    output logic [NUM_REGISTERS-1:0] register_write_strobe
);

    // AXI-Lite states
    typedef enum logic [2:0] {
        AXI_IDLE,
        AXI_WRITE_ADDR,
        AXI_WRITE_DATA,
        AXI_WRITE_RESP,
        AXI_READ_ADDR,
        AXI_READ_DATA
    } axi_state_t;
    
    axi_state_t current_state, next_state;
    
    // Internal registers
    logic [ADDR_WIDTH-1:0] write_addr, read_addr;
    logic [DATA_WIDTH-1:0] write_data;
    logic [$clog2(NUM_REGISTERS)-1:0] write_reg_index, read_reg_index;
    
    // AXI-Lite state machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= AXI_IDLE;
            write_addr <= '0;
            read_addr <= '0;
            write_data <= '0;
            write_reg_index <= '0;
            read_reg_index <= '0;
            
            // Reset AXI-Lite interface signals
            s_axil_awready <= 1'b0;
            s_axil_wready <= 1'b0;
            s_axil_bvalid <= 1'b0;
            s_axil_bresp <= 2'b00;
            s_axil_arready <= 1'b0;
            s_axil_rvalid <= 1'b0;
            s_axil_rdata <= '0;
            s_axil_rresp <= 2'b00;
            
            // Reset register interface signals
            for (int i = 0; i < NUM_REGISTERS; i++) begin
                registers_out[i] <= '0;
                register_write_strobe[i] <= 1'b0;
            end
        end else begin
            current_state <= next_state;
            
            // Default values for register write strobe
            for (int i = 0; i < NUM_REGISTERS; i++) begin
                register_write_strobe[i] <= 1'b0;
            end
            
            case (current_state)
                AXI_IDLE: begin
                    // Default state with interface signals set for new transactions
                    s_axil_awready <= 1'b1;
                    s_axil_arready <= 1'b1;
                    s_axil_wready <= 1'b0;
                    s_axil_bvalid <= 1'b0;
                    s_axil_rvalid <= 1'b0;
                    
                    // Capture write address
                    if (s_axil_awvalid && s_axil_awready) begin
                        write_addr <= s_axil_awaddr;
                        write_reg_index <= s_axil_awaddr[$clog2(NUM_REGISTERS)+1:2]; // Assumes 32-bit registers
                        s_axil_awready <= 1'b0;
                    end
                    
                    // Capture read address
                    if (s_axil_arvalid && s_axil_arready) begin
                        read_addr <= s_axil_araddr;
                        read_reg_index <= s_axil_araddr[$clog2(NUM_REGISTERS)+1:2]; // Assumes 32-bit registers
                        s_axil_arready <= 1'b0;
                    end
                end
                
                AXI_WRITE_ADDR: begin
                    s_axil_wready <= 1'b1;
                    
                    // Capture write data
                    if (s_axil_wvalid && s_axil_wready) begin
                        write_data <= s_axil_wdata;
                        s_axil_wready <= 1'b0;
                    end
                end
                
                AXI_WRITE_DATA: begin
                    // Write data to register
                    if (write_reg_index < NUM_REGISTERS) begin
                        registers_out[write_reg_index] <= write_data;
                        register_write_strobe[write_reg_index] <= 1'b1;
                    end
                    
                    // Prepare write response
                    s_axil_bvalid <= 1'b1;
                    s_axil_bresp <= 2'b00;  // OKAY
                end
                
                AXI_WRITE_RESP: begin
                    // Wait for write response to be accepted
                    if (s_axil_bready) begin
                        s_axil_bvalid <= 1'b0;
                    end
                end
                
                AXI_READ_ADDR: begin
                    // Prepare read data
                    if (read_reg_index < NUM_REGISTERS) begin
                        s_axil_rdata <= registers_in[read_reg_index];
                        s_axil_rresp <= 2'b00;  // OKAY
                    end else begin
                        s_axil_rdata <= 32'h0;
                        s_axil_rresp <= 2'b10;  // SLVERR
                    end
                    
                    s_axil_rvalid <= 1'b1;
                end
                
                AXI_READ_DATA: begin
                    // Wait for read data to be accepted
                    if (s_axil_rready) begin
                        s_axil_rvalid <= 1'b0;
                    end
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            AXI_IDLE: begin
                if (s_axil_awvalid && s_axil_awready) begin
                    next_state = AXI_WRITE_ADDR;
                end else if (s_axil_arvalid && s_axil_arready) begin
                    next_state = AXI_READ_ADDR;
                end
            end
            
            AXI_WRITE_ADDR: begin
                if (s_axil_wvalid && s_axil_wready) begin
                    next_state = AXI_WRITE_DATA;
                end
            end
            
            AXI_WRITE_DATA: begin
                next_state = AXI_WRITE_RESP;
            end
            
            AXI_WRITE_RESP: begin
                if (s_axil_bready && s_axil_bvalid) begin
                    next_state = AXI_IDLE;
                end
            end
            
            AXI_READ_ADDR: begin
                next_state = AXI_READ_DATA;
            end
            
            AXI_READ_DATA: begin
                if (s_axil_rready && s_axil_rvalid) begin
                    next_state = AXI_IDLE;
                end
            end
            
            default: next_state = AXI_IDLE;
        endcase
    end

endmodule
