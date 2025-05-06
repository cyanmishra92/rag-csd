//-----------------------------------------------------------------------------
// File: mean_pooler.sv
// 
// Description: Mean Pooler Module for RAG-CSD
//              Performs mean pooling over token embeddings
//
// Parameters:
//   EMBEDDING_DIM  - Embedding dimension
//   MAX_TOKENS     - Maximum tokens per query
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module mean_pooler #(
    parameter int EMBEDDING_DIM = 384,
    parameter int MAX_TOKENS = 128
) (
    // Clock and reset
    input  logic                                          clk,
    input  logic                                          rst_n,
    
    // Control signals
    input  logic                                          start,
    output logic                                          done,
    
    // Sequence info
    input  logic [$clog2(MAX_TOKENS):0]                   sequence_length,
    
    // Input token embeddings and output pooled embedding
    input  logic [MAX_TOKENS-1:0][EMBEDDING_DIM-1:0][31:0]  token_embeddings,
    output logic [EMBEDDING_DIM-1:0][31:0]                  pooled_embedding
);

    // Pooler states
    typedef enum logic [2:0] {
        MP_IDLE,
        MP_SUM,
        MP_DIVIDE,
        MP_FINISHED
    } mp_state_t;
    
    mp_state_t current_state, next_state;
    
    // Internal registers
    logic [$clog2(EMBEDDING_DIM):0] dim_idx;
    logic [31:0] sequence_length_float;  // For floating-point division
    logic [31:0] embedding_sums [EMBEDDING_DIM-1:0];
    
    // State machine logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= MP_IDLE;
            dim_idx <= '0;
            done <= 1'b0;
            
            // Initialize sums
            for (int i = 0; i < EMBEDDING_DIM; i++) begin
                embedding_sums[i] <= '0;
                pooled_embedding[i] <= '0;
            end
        end else begin
            current_state <= next_state;
            
            case (current_state)
                MP_IDLE: begin
                    if (start) begin
                        dim_idx <= '0;
                        done <= 1'b0;
                        
                        // Convert sequence_length to float for later division
                        sequence_length_float <= $shortrealtobits(sequence_length);
                        
                        // Reset sums
                        for (int i = 0; i < EMBEDDING_DIM; i++) begin
                            embedding_sums[i] <= '0;
                        end
                    end
                end
                
                MP_SUM: begin
                    // Sum across all tokens for each dimension
                    if (dim_idx < EMBEDDING_DIM) begin
                        // Clear the sum for this dimension
                        embedding_sums[dim_idx] <= '0;
                        
                        // Accumulate sum across all tokens
                        for (int t = 0; t < sequence_length; t++) begin
                            embedding_sums[dim_idx] <= embedding_sums[dim_idx] + token_embeddings[t][dim_idx];
                        end
                        
                        dim_idx <= dim_idx + 1;
                    end
                end
                
                MP_DIVIDE: begin
                    // Divide sums by sequence length to get means
                    for (int i = 0; i < EMBEDDING_DIM; i++) begin
                        pooled_embedding[i] <= embedding_sums[i] / sequence_length_float;
                    end
                end
                
                MP_FINISHED: begin
                    done <= 1'b1;
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            MP_IDLE: begin
                if (start) begin
                    next_state = MP_SUM;
                end
            end
            
            MP_SUM: begin
                if (dim_idx >= EMBEDDING_DIM) begin
                    next_state = MP_DIVIDE;
                end
            end
            
            MP_DIVIDE: begin
                next_state = MP_FINISHED;
            end
            
            MP_FINISHED: begin
                next_state = MP_IDLE;
            end
            
            default: next_state = MP_IDLE;
        endcase
    end

endmodule
