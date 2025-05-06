//-----------------------------------------------------------------------------
// File: axi_stream_interface.sv
// 
// Description: AXI-Stream Interface Module for RAG-CSD
//              Handles data streaming between components
//
// Parameters:
//   DATA_WIDTH     - Data width
//   ID_WIDTH       - ID width
//   DEST_WIDTH     - Destination width
//   USER_WIDTH     - User width
//   FIFO_DEPTH     - FIFO depth for buffering
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module axi_stream_interface #(
    parameter int DATA_WIDTH = 512,
    parameter int ID_WIDTH = 4,
    parameter int DEST_WIDTH = 4,
    parameter int USER_WIDTH = 4,
    parameter int FIFO_DEPTH = 16
) (
    // Clock and reset
    input  logic                   clk,
    input  logic                   rst_n,
    
    // Slave side (input)
    input  logic                   s_axis_tvalid,
    output logic                   s_axis_tready,
    input  logic [DATA_WIDTH-1:0]  s_axis_tdata,
    input  logic                   s_axis_tlast,
    input  logic [ID_WIDTH-1:0]    s_axis_tid,
    input  logic [DEST_WIDTH-1:0]  s_axis_tdest,
    input  logic [USER_WIDTH-1:0]  s_axis_tuser,
    
    // Master side (output)
    output logic                   m_axis_tvalid,
    input  logic                   m_axis_tready,
    output logic [DATA_WIDTH-1:0]  m_axis_tdata,
    output logic                   m_axis_tlast,
    output logic [ID_WIDTH-1:0]    m_axis_tid,
    output logic [DEST_WIDTH-1:0]  m_axis_tdest,
    output logic [USER_WIDTH-1:0]  m_axis_tuser,
    
    // Status signals
    output logic                   fifo_full,
    output logic                   fifo_empty,
    output logic [$clog2(FIFO_DEPTH):0] fifo_count
);

    // FIFO structure for AXI-Stream data
    typedef struct packed {
        logic [DATA_WIDTH-1:0] tdata;
        logic tlast;
        logic [ID_WIDTH-1:0] tid;
        logic [DEST_WIDTH-1:0] tdest;
        logic [USER_WIDTH-1:0] tuser;
    } axis_packet_t;
    
    // FIFO storage
    axis_packet_t fifo [FIFO_DEPTH-1:0];
    logic [$clog2(FIFO_DEPTH)-1:0] wr_ptr, rd_ptr;
    logic [$clog2(FIFO_DEPTH):0] count;
    
    // FIFO control
    assign fifo_full = (count == FIFO_DEPTH);
    assign fifo_empty = (count == 0);
    assign fifo_count = count;
    
    // Set slave ready when FIFO not full
    assign s_axis_tready = !fifo_full;
    
    // Set master valid when FIFO not empty
    assign m_axis_tvalid = !fifo_empty;
    
    // Output data from FIFO
    assign m_axis_tdata = fifo[rd_ptr].tdata;
    assign m_axis_tlast = fifo[rd_ptr].tlast;
    assign m_axis_tid = fifo[rd_ptr].tid;
    assign m_axis_tdest = fifo[rd_ptr].tdest;
    assign m_axis_tuser = fifo[rd_ptr].tuser;
    
    // FIFO write and read logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= '0;
            rd_ptr <= '0;
            count <= '0;
            
            // Clear FIFO
            for (int i = 0; i < FIFO_DEPTH; i++) begin
                fifo[i].tdata <= '0;
                fifo[i].tlast <= 1'b0;
                fifo[i].tid <= '0;
                fifo[i].tdest <= '0;
                fifo[i].tuser <= '0;
            end
        end else begin
            // Write to FIFO
            if (s_axis_tvalid && s_axis_tready) begin
                fifo[wr_ptr].tdata <= s_axis_tdata;
                fifo[wr_ptr].tlast <= s_axis_tlast;
                fifo[wr_ptr].tid <= s_axis_tid;
                fifo[wr_ptr].tdest <= s_axis_tdest;
                fifo[wr_ptr].tuser <= s_axis_tuser;
                
                // Update write pointer
                if (wr_ptr == FIFO_DEPTH - 1) begin
                    wr_ptr <= '0;
                end else begin
                    wr_ptr <= wr_ptr + 1;
                end
            end
            
            // Read from FIFO
            if (m_axis_tvalid && m_axis_tready) begin
                // Update read pointer
                if (rd_ptr == FIFO_DEPTH - 1) begin
                    rd_ptr <= '0;
                end else begin
                    rd_ptr <= rd_ptr + 1;
                end
            end
            
            // Update count
            if ((s_axis_tvalid && s_axis_tready) && !(m_axis_tvalid && m_axis_tready)) begin
                // Write only
                count <= count + 1;
            end else if (!(s_axis_tvalid && s_axis_tready) && (m_axis_tvalid && m_axis_tready)) begin
                // Read only
                count <= count - 1;
            end
            // Both write and read or neither - count unchanged
        end
    end

endmodule
