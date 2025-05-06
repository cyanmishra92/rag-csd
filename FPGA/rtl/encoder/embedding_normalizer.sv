//-----------------------------------------------------------------------------
// File: embedding_normalizer.sv
// 
// Description: Embedding Normalizer Module for RAG-CSD
//              Normalizes embeddings to unit length (L2 norm)
//
// Parameters:
//   EMBEDDING_DIM  - Embedding dimension
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module embedding_normalizer #(
    parameter int EMBEDDING_DIM = 384
) (
    // Clock and reset
    input  logic                        clk,
    input  logic                        rst_n,
    
    // Control signals
    input  logic                        start,
    output logic                        done,
    
    // Input and output embeddings
    input  logic [EMBEDDING_DIM-1:0][31:0]  input_embedding,
    output logic [EMBEDDING_DIM-1:0][31:0]  normalized_embedding
);

    // Normalizer states
    typedef enum logic [2:0] {
        N_IDLE,
        N_COMPUTE_SQUARE_SUM,
        N_COMPUTE_SQRT,
        N_NORMALIZE,
        N_FINISHED
    } norm_state_t;
    
    norm_state_t current_state, next_state;
    
    // Internal registers
    logic [31:0] square_sum;
    logic [31:0] norm;
    logic [31:0] epsilon;
    
    // Constants
    localparam logic [31:0] EPSILON = 32'h3a83126f; // 0.001f in IEEE-754
    
    // State machine logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= N_IDLE;
            square_sum <= '0;
            norm <= '0;
            epsilon <= EPSILON;
            done <= 1'b0;
        end else begin
            current_state <= next_state;
            
            case (current_state)
                N_IDLE: begin
                    if (start) begin
                        square_sum <= '0;
                        done <= 1'b0;
                    end
                end
                
                N_COMPUTE_SQUARE_SUM: begin
                    // Compute sum of squares
                    square_sum <= '0;
                    
                    for (int i = 0; i < EMBEDDING_DIM; i++) begin
                        square_sum <= square_sum + (input_embedding[i] * input_embedding[i]);
                    end
                end
                
                N_COMPUTE_SQRT: begin
                    // Compute square root of sum plus epsilon
                    // In real hardware, would use an optimized CORDIC algorithm or lookup table
                    norm <= $sqrt(square_sum + epsilon);
                end
                
                N_NORMALIZE: begin
                    // Divide each element by the norm
                    if (norm > 0) begin
                        for (int i = 0; i < EMBEDDING_DIM; i++) begin
                            normalized_embedding[i] <= input_embedding[i] / norm;
                        end
                    end else begin
                        // If norm is zero, return input as is
                        for (int i = 0; i < EMBEDDING_DIM; i++) begin
                            normalized_embedding[i] <= input_embedding[i];
                        end
                    end
                end
                
                N_FINISHED: begin
                    done <= 1'b1;
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            N_IDLE: begin
                if (start) begin
                    next_state = N_COMPUTE_SQUARE_SUM;
                end
            end
            
            N_COMPUTE_SQUARE_SUM: begin
                next_state = N_COMPUTE_SQRT;
            end
            
            N_COMPUTE_SQRT: begin
                next_state = N_NORMALIZE;
            end
            
            N_NORMALIZE: begin
                next_state = N_FINISHED;
            end
            
            N_FINISHED: begin
                next_state = N_IDLE;
            end
            
            default: next_state = N_IDLE;
        endcase
    end

endmodule
