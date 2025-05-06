//-----------------------------------------------------------------------------
// File: text_buffer_manager.sv
// 
// Description: Text Buffer Manager Module for RAG-CSD
//              Handles text buffer operations for augmentation
//
// Parameters:
//   MAX_SEQUENCE_LEN  - Maximum sequence length
//   TOP_K             - Number of retrieved documents
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module text_buffer_manager #(
    parameter int MAX_SEQUENCE_LEN = 512,
    parameter int TOP_K = 5
) (
    // Clock and reset
    input  logic                            clk,
    input  logic                            rst_n,
    
    // Control signals
    input  logic                            start,
    output logic                            done,
    
    // Augmentation strategy
    input  logic [1:0]                      strategy,
    
    // Input buffers
    input  logic [7:0] query_buffer [MAX_SEQUENCE_LEN-1:0],
    input  logic [31:0] query_length,
    input  logic [7:0] doc_buffers [TOP_K-1:0][MAX_SEQUENCE_LEN-1:0],
    input  logic [31:0] doc_lengths [TOP_K-1:0],
    input  logic [TOP_K-1:0] doc_included,
    input  logic [TOP_K-1:0][31:0] similarity_scores,
    
    // Output buffer
    output logic [7:0] output_buffer [MAX_SEQUENCE_LEN*(TOP_K+1)-1:0],
    output logic [31:0] output_length
);

    // Buffer manager states
    typedef enum logic [2:0] {
        BM_IDLE,
        BM_COPY_QUERY,
        BM_ADD_SEPARATOR,
        BM_COPY_DOCS,
        BM_FINALIZE,
        BM_FINISHED
    } buffer_state_t;
    
    buffer_state_t current_state, next_state;
    
    // Internal registers
    logic [31:0] output_pos;
    logic [$clog2(TOP_K)-1:0] doc_idx;
    logic [31:0] doc_pos;
    
    // Separator patterns
    logic [7:0] newline_separator [1:0];
    
    // Initialize separators
    initial begin
        // "\n\n"
        newline_separator[0] = 8'h0A; // \n
        newline_separator[1] = 8'h0A; // \n
    end
    
    // State machine logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= BM_IDLE;
            output_pos <= '0;
            doc_idx <= '0;
            doc_pos <= '0;
            output_length <= '0;
            done <= 1'b0;
            
            // Initialize output buffer
            for (int i = 0; i < MAX_SEQUENCE_LEN*(TOP_K+1); i++) begin
                output_buffer[i] <= 8'h00;
            end
        end else begin
            current_state <= next_state;
            
            case (current_state)
                BM_IDLE: begin
                    if (start) begin
                        output_pos <= '0;
                        doc_idx <= '0;
                        doc_pos <= '0;
                        done <= 1'b0;
                    end
                end
                
                BM_COPY_QUERY: begin
                    // Copy query to output buffer
                    if (output_pos < query_length) begin
                        output_buffer[output_pos] <= query_buffer[output_pos];
                        output_pos <= output_pos + 1;
                    end
                end
                
                BM_ADD_SEPARATOR: begin
                    // Add separator between query and documents
                    if (output_pos < query_length + 2) begin
                        output_buffer[output_pos] <= newline_separator[output_pos - query_length];
                        output_pos <= output_pos + 1;
                    end
                end
                
                BM_COPY_DOCS: begin
                    // Copy documents to output buffer
                    if (doc_idx < TOP_K) begin
                        if (doc_included[doc_idx]) begin
                            if (doc_pos < doc_lengths[doc_idx]) begin
                                // Copy document content
                                output_buffer[output_pos] <= doc_buffers[doc_idx][doc_pos];
                                output_pos <= output_pos + 1;
                                doc_pos <= doc_pos + 1;
                            end else if (doc_idx < TOP_K - 1) begin
                                // Add separator between documents
                                if (doc_pos < doc_lengths[doc_idx] + 2) begin
                                    output_buffer[output_pos] <= newline_separator[doc_pos - doc_lengths[doc_idx]];
                                    output_pos <= output_pos + 1;
                                    doc_pos <= doc_pos + 1;
                                end else begin
                                    // Move to next document
                                    doc_idx <= doc_idx + 1;
                                    doc_pos <= '0;
                                }
                            end else begin
                                // Last document, move to finalize
                                doc_idx <= doc_idx + 1;
                            end
                        end else begin
                            // Skip excluded document
                            doc_idx <= doc_idx + 1;
                        end
                    end
                end
                
                BM_FINALIZE: begin
                    // Set final output length
                    output_length <= output_pos;
                end
                
                BM_FINISHED: begin
                    done <= 1'b1;
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            BM_IDLE: begin
                if (start) begin
                    next_state = BM_COPY_QUERY;
                end
            end
            
            BM_COPY_QUERY: begin
                if (output_pos >= query_length) begin
                    next_state = BM_ADD_SEPARATOR;
                end
            end
            
            BM_ADD_SEPARATOR: begin
                if (output_pos >= query_length + 2) begin
                    next_state = BM_COPY_DOCS;
                end
            end
            
            BM_COPY_DOCS: begin
                if (doc_idx >= TOP_K) begin
                    next_state = BM_FINALIZE;
                end
            end
            
            BM_FINALIZE: begin
                next_state = BM_FINISHED;
            end
            
            BM_FINISHED: begin
                next_state = BM_IDLE;
            end
            
            default: next_state = BM_IDLE;
        endcase
    end

endmodule
