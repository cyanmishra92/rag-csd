//-----------------------------------------------------------------------------
// File: top_k_priority_queue.sv
// 
// Description: Top-K Priority Queue Module for RAG-CSD
//              Maintains the top-k results by similarity score
//
// Parameters:
//   K              - Number of top results to maintain
//   EMBEDDING_DIM  - Embedding dimension
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module top_k_priority_queue #(
    parameter int K = 5,
    parameter int EMBEDDING_DIM = 384
) (
    // Clock and reset
    input  logic                                    clk,
    input  logic                                    rst_n,
    
    // Control signals
    input  logic                                    push,
    output logic                                    done,
    
    // Input data
    input  logic [31:0]                             similarity,
    input  logic [31:0]                             doc_index,
    input  logic [EMBEDDING_DIM-1:0][31:0]          vector,
    
    // Output top-k results
    output logic [K-1:0][31:0]                      topk_similarities,
    output logic [K-1:0][31:0]                      topk_indices,
    output logic [K-1:0][EMBEDDING_DIM-1:0][31:0]   topk_vectors
);

    // Priority queue states
    typedef enum logic [2:0] {
        PQ_IDLE,
        PQ_FIND_POSITION,
        PQ_SHIFT_ELEMENTS,
        PQ_INSERT,
        PQ_FINISHED
    } pq_state_t;
    
    pq_state_t current_state, next_state;
    
    // Internal registers
    logic [$clog2(K):0] position;       // Position to insert new element
    logic [$clog2(K):0] shift_idx;      // Current index for shifting
    logic insertion_needed;             // Flag indicating if insertion is needed
    
    // State machine logic with optimized shifting
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= PQ_IDLE;
            position <= K;  // Default position is outside the array
            shift_idx <= '0;
            insertion_needed <= 1'b0;
            done <= 1'b0;
            
            // Initialize top-k arrays
            for (int i = 0; i < K; i++) begin
                topk_similarities[i] <= '0;
                topk_indices[i] <= '0;
            end
        end else begin
            current_state <= next_state;
            
            case (current_state)
                PQ_IDLE: begin
                    if (push) begin
                        position <= K;  // Default position is outside the array
                        shift_idx <= K - 1;
                        insertion_needed <= 1'b0;
                        done <= 1'b0;
                    end
                end
                
                PQ_FIND_POSITION: begin
                    // Find position to insert new element (optimized)
                    insertion_needed <= 1'b0;
                    
                    // Binary search would be more efficient for large K
                    // For small K, linear search is simpler
                    for (int i = 0; i < K; i++) begin
                        if (similarity > topk_similarities[i]) begin
                            position <= i;
                            insertion_needed <= 1'b1;
                            break;
                        end
                    end
                end
                
                PQ_SHIFT_ELEMENTS: begin
                    if (insertion_needed) begin
                        // Shift elements down to make space, starting from the bottom
                        if (shift_idx > position) begin
                            // Shift this element down
                            topk_similarities[shift_idx] <= topk_similarities[shift_idx-1];
                            topk_indices[shift_idx] <= topk_indices[shift_idx-1];
                            topk_vectors[shift_idx] <= topk_vectors[shift_idx-1];
                            
                            shift_idx <= shift_idx - 1;
                        end
                    end
                end
                
                PQ_INSERT: begin
                    if (insertion_needed) begin
                        // Insert the new element
                        topk_similarities[position] <= similarity;
                        topk_indices[position] <= doc_index;
                        topk_vectors[position] <= vector;
                    end
                end
                
                PQ_FINISHED: begin
                    done <= 1'b1;
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            PQ_IDLE: begin
                if (push) begin
                    next_state = PQ_FIND_POSITION;
                end
            end
            
            PQ_FIND_POSITION: begin
                next_state = PQ_SHIFT_ELEMENTS;
            end
            
            PQ_SHIFT_ELEMENTS: begin
                if (!insertion_needed || shift_idx <= position) begin
                    next_state = PQ_INSERT;
                end
            end
            
            PQ_INSERT: begin
                next_state = PQ_FINISHED;
            end
            
            PQ_FINISHED: begin
                next_state = PQ_IDLE;
            end
            
            default: next_state = PQ_IDLE;
        endcase
    end

endmodule
