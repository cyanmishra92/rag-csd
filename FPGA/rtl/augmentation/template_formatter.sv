//-----------------------------------------------------------------------------
// File: template_formatter.sv
// 
// Description: Template Formatter Module for RAG-CSD
//              Formats query and documents using predefined templates
//
// Parameters:
//   MAX_SEQUENCE_LEN  - Maximum sequence length
//   TOP_K             - Number of retrieved documents
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module template_formatter #(
    parameter int MAX_SEQUENCE_LEN = 512,
    parameter int TOP_K = 5
) (
    // Clock and reset
    input  logic                            clk,
    input  logic                            rst_n,
    
    // Control signals
    input  logic                            start,
    output logic                            done,
    
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

    // Template formatter states
    typedef enum logic [3:0] {
        TF_IDLE,
        TF_ADD_QUERY_PREFIX,
        TF_COPY_QUERY,
        TF_ADD_SEPARATOR,
        TF_ADD_CONTEXT_PREFIX,
        TF_COPY_DOCS,
        TF_ADD_DOC_SEPARATOR,
        TF_FINALIZE,
        TF_FINISHED
    } template_state_t;
    
    template_state_t current_state, next_state;
    
    // Internal registers
    logic [31:0] output_pos;
    logic [$clog2(TOP_K)-1:0] doc_idx;
    logic [31:0] doc_pos;
    logic [31:0] prefix_pos;
    
    // Template prefixes and separators
    logic [7:0] query_prefix [6:0];    // "Query: "
    logic [7:0] context_prefix [8:0];  // "Context: "
    logic [7:0] newline_separator [1:0];
    
    // Initialize template strings
    initial begin
        // "Query: "
        query_prefix[0] = "Q";
        query_prefix[1] = "u";
        query_prefix[2] = "e";
        query_prefix[3] = "r";
        query_prefix[4] = "y";
        query_prefix[5] = ":";
        query_prefix[6] = " ";
        
        // "Context: "
        context_prefix[0] = "C";
        context_prefix[1] = "o";
        context_prefix[2] = "n";
        context_prefix[3] = "t";
        context_prefix[4] = "e";
        context_prefix[5] = "x";
        context_prefix[6] = "t";
        context_prefix[7] = ":";
        context_prefix[8] = " ";
        
        // "\n\n"
        newline_separator[0] = 8'h0A; // \n
        newline_separator[1] = 8'h0A; // \n
    end
    
    // State machine logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= TF_IDLE;
            output_pos <= '0;
            doc_idx <= '0;
            doc_pos <= '0;
            prefix_pos <= '0;
            output_length <= '0;
            done <= 1'b0;
            
            // Initialize output buffer
            for (int i = 0; i < MAX_SEQUENCE_LEN*(TOP_K+1); i++) begin
                output_buffer[i] <= 8'h00;
            end
        end else begin
            current_state <= next_state;
            
            case (current_state)
                TF_IDLE: begin
                    if (start) begin
                        output_pos <= '0;
                        doc_idx <= '0;
                        doc_pos <= '0;
                        prefix_pos <= '0;
                        done <= 1'b0;
                    end
                end
                
                TF_ADD_QUERY_PREFIX: begin
                    // Add "Query: " prefix
                    if (prefix_pos < 7) begin
                        output_buffer[output_pos] <= query_prefix[prefix_pos];
                        output_pos <= output_pos + 1;
                        prefix_pos <= prefix_pos + 1;
                    end
                end
                
                TF_COPY_QUERY: begin
                    // Copy query text
                    if (doc_pos < query_length) begin
                        output_buffer[output_pos] <= query_buffer[doc_pos];
                        output_pos <= output_pos + 1;
                        doc_pos <= doc_pos + 1;
                    end
                end
                
                TF_ADD_SEPARATOR: begin
                    // Add separator between query and context
                    if (prefix_pos < 2) begin
                        output_buffer[output_pos] <= newline_separator[prefix_pos];
                        output_pos <= output_pos + 1;
                        prefix_pos <= prefix_pos + 1;
                    end
                end
                
                TF_ADD_CONTEXT_PREFIX: begin
                    // Add "Context: " prefix
                    if (prefix_pos < 9) begin
                        output_buffer[output_pos] <= context_prefix[prefix_pos];
                        output_pos <= output_pos + 1;
                        prefix_pos <= prefix_pos + 1;
                    end
                end
                
                TF_COPY_DOCS: begin
                    // Copy document content
                    if (doc_idx < TOP_K && doc_included[doc_idx]) begin
                        if (doc_pos < doc_lengths[doc_idx]) begin
                            output_buffer[output_pos] <= doc_buffers[doc_idx][doc_pos];
                            output_pos <= output_pos + 1;
                            doc_pos <= doc_pos + 1;
                        end
                    end
                end
                
                TF_ADD_DOC_SEPARATOR: begin
                    // Add separator between documents if not the last one
                    if (doc_idx < TOP_K - 1) begin
                        // Find next included document
                        logic next_included = 1'b0;
                        for (int i = doc_idx + 1; i < TOP_K; i++) begin
                            if (doc_included[i]) begin
                                next_included = 1'b1;
                                break;
                            end
                        end
                        
                        if (next_included) begin
                            if (prefix_pos < 2) begin
                                output_buffer[output_pos] <= newline_separator[prefix_pos];
                                output_pos <= output_pos + 1;
                                prefix_pos <= prefix_pos + 1;
                            end
                        end
                    end
                    
                    // Prepare to process next document
                    doc_idx <= doc_idx + 1;
                    doc_pos <= '0;
                    prefix_pos <= '0;
                end
                
                TF_FINALIZE: begin
                    // Set final output length
                    output_length <= output_pos;
                end
                
                TF_FINISHED: begin
                    done <= 1'b1;
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            TF_IDLE: begin
                if (start) begin
                    next_state = TF_ADD_QUERY_PREFIX;
                end
            end
            
            TF_ADD_QUERY_PREFIX: begin
                if (prefix_pos >= 7) begin
                    next_state = TF_COPY_QUERY;
                    prefix_pos = '0;
                    doc_pos = '0;
                end
            end
            
            TF_COPY_QUERY: begin
                if (doc_pos >= query_length) begin
                    next_state = TF_ADD_SEPARATOR;
                    prefix_pos = '0;
                end
            end
            
            TF_ADD_SEPARATOR: begin
                if (prefix_pos >= 2) begin
                    next_state = TF_ADD_CONTEXT_PREFIX;
                    prefix_pos = '0;
                end
            end
            
            TF_ADD_CONTEXT_PREFIX: begin
                if (prefix_pos >= 9) begin
                    next_state = TF_COPY_DOCS;
                    doc_pos = '0;
                end
            end
            
            TF_COPY_DOCS: begin
                if (doc_idx >= TOP_K) begin
                    next_state = TF_FINALIZE;
                end else if (doc_pos >= doc_lengths[doc_idx] || !doc_included[doc_idx]) begin
                    next_state = TF_ADD_DOC_SEPARATOR;
                    prefix_pos = '0;
                end
            end
            
            TF_ADD_DOC_SEPARATOR: begin
                if (doc_idx >= TOP_K - 1 || prefix_pos >= 2) begin
                    next_state = TF_COPY_DOCS;
                end
            end
            
            TF_FINALIZE: begin
                next_state = TF_FINISHED;
            end
            
            TF_FINISHED: begin
                next_state = TF_IDLE;
            end
            
            default: next_state = TF_IDLE;
        endcase
    end

endmodule
