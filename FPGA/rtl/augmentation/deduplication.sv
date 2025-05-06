//-----------------------------------------------------------------------------
// File: deduplication.sv
// 
// Description: Deduplication Module for RAG-CSD
//              Removes duplicate content from retrieved documents
//
// Parameters:
//   MAX_SEQUENCE_LEN  - Maximum sequence length
//   TOP_K             - Number of retrieved documents
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module deduplication #(
    parameter int MAX_SEQUENCE_LEN = 512,
    parameter int TOP_K = 5
) (
    // Clock and reset
    input  logic                            clk,
    input  logic                            rst_n,
    
    // Control signals
    input  logic                            start,
    output logic                            done,
    
    // Document content
    input  logic [7:0] doc_buffers [TOP_K-1:0][MAX_SEQUENCE_LEN-1:0],
    input  logic [31:0] doc_lengths [TOP_K-1:0],
    output logic [TOP_K-1:0] doc_included
);

    // Deduplication states
    typedef enum logic [2:0] {
        D_IDLE,
        D_COMPARE_DOCS,
        D_MARK_DUPLICATES,
        D_FINISHED
    } dedup_state_t;
    
    dedup_state_t current_state, next_state;
    
    // Internal registers
    logic [$clog2(TOP_K)-1:0] doc_i, doc_j;
    logic [TOP_K-1:0][TOP_K-1:0] similarity_matrix;
    logic [31:0] similarity_threshold;
    
    // Similarity threshold (80% similarity considered duplicate)
    localparam logic [31:0] DEFAULT_THRESHOLD = 32'h3f4ccccd; // 0.8f in IEEE-754
    
    // State machine logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= D_IDLE;
            doc_i <= '0;
            doc_j <= '0;
            similarity_threshold <= DEFAULT_THRESHOLD;
            done <= 1'b0;
            
            // Initialize matrices
            for (int i = 0; i < TOP_K; i++) begin
                for (int j = 0; j < TOP_K; j++) begin
                    similarity_matrix[i][j] <= 1'b0;
                end
            end
        end else begin
            current_state <= next_state;
            
            case (current_state)
                D_IDLE: begin
                    if (start) begin
                        doc_i <= '0;
                        doc_j <= '0;
                        done <= 1'b0;
                        
                        // Clear similarity matrix
                        for (int i = 0; i < TOP_K; i++) begin
                            for (int j = 0; j < TOP_K; j++) begin
                                similarity_matrix[i][j] <= 1'b0;
                            end
                        end
                    end
                end
                
                D_COMPARE_DOCS: begin
                    // Compare pairs of documents for similarity
                    if (doc_i < TOP_K && doc_j < TOP_K) begin
                        if (doc_i != doc_j && doc_included[doc_i] && doc_included[doc_j]) begin
                            // Compute similarity based on content overlap
                            logic [31:0] match_count = 0;
                            logic [31:0] total_count = doc_lengths[doc_i];
                            
                            // Simple n-gram based similarity
                            for (int offset = 0; offset < doc_lengths[doc_i] - 10; offset++) begin
                                logic is_match = 1'b1;
                                
                                // Check for 10-gram match
                                for (int k = 0; k < 10; k++) begin
                                    if (doc_buffers[doc_i][offset + k] != doc_buffers[doc_j][offset + k]) begin
                                        is_match = 1'b0;
                                        break;
                                    end
                                end
                                
                                if (is_match) begin
                                    match_count = match_count + 10;
                                    offset = offset + 10; // Skip ahead
                                end
                            end
                            
                            // Calculate similarity
                            logic [31:0] similarity = match_count * 32'h3f800000 / total_count; // match/total as float
                            
                            // Mark as similar if above threshold
                            if (similarity >= similarity_threshold) begin
                                similarity_matrix[doc_i][doc_j] <= 1'b1;
                            end
                        end
                        
                        // Move to next pair
                        if (doc_j < TOP_K - 1) begin
                            doc_j <= doc_j + 1;
                        end else begin
                            doc_j <= '0;
                            doc_i <= doc_i + 1;
                        end
                    end
                end
                
                D_MARK_DUPLICATES: begin
                    // For each document, if it has a similar document with higher score, exclude it
                    for (int i = 0; i < TOP_K; i++) begin
                        for (int j = 0; j < TOP_K; j++) begin
                            if (similarity_matrix[i][j] == 1'b1) begin
                                // Exclude the document with lower similarity score (larger index usually)
                                doc_included[j] <= 1'b0;
                            end
                        end
                    end
                end
                
                D_FINISHED: begin
                    done <= 1'b1;
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            D_IDLE: begin
                if (start) begin
                    next_state = D_COMPARE_DOCS;
                end
            end
            
            D_COMPARE_DOCS: begin
                if (doc_i >= TOP_K - 1 && doc_j >= TOP_K - 1) begin
                    next_state = D_MARK_DUPLICATES;
                end
            end
            
            D_MARK_DUPLICATES: begin
                next_state = D_FINISHED;
            end
            
            D_FINISHED: begin
                next_state = D_IDLE;
            end
            
            default: next_state = D_IDLE;
        endcase
    end

endmodule
