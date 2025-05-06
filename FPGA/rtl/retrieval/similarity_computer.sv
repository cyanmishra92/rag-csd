//-----------------------------------------------------------------------------
// File: similarity_computer.sv
// 
// Description: Similarity Computer Module for RAG-CSD
//              Computes similarity between two vectors using different metrics
//
// Parameters:
//   EMBEDDING_DIM  - Embedding dimension
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module similarity_computer #(
    parameter int EMBEDDING_DIM = 384
) (
    // Clock and reset
    input  logic                            clk,
    input  logic                            rst_n,
    
    // Control signals
    input  logic                            start,
    output logic                            done,
    
    // Similarity metric
    input  logic [1:0]                      metric, // 0=cosine, 1=dot, 2=euclidean
    
    // Input vectors
    input  logic [EMBEDDING_DIM-1:0][31:0]  vec_a,
    input  logic [EMBEDDING_DIM-1:0][31:0]  vec_b,
    
    // Output similarity
    output logic [31:0]                     similarity
);

    // Similarity computation states
    typedef enum logic [3:0] {
        SC_IDLE,
        SC_DOT_PRODUCT,
        SC_COMPUTE_NORMS,
        SC_COSINE,
        SC_EUCLIDEAN,
        SC_FINISHED
    } sim_state_t;
    
    sim_state_t current_state, next_state;
    
    // Internal registers
    logic [31:0] dot_product;
    logic [31:0] norm_a, norm_b;
    logic [31:0] distance_sq;
    logic [31:0] epsilon;
    
    // Constants
    localparam logic [31:0] EPSILON = 32'h3a83126f; // 0.001f in IEEE-754
    
    // Pipeline registers for parallelization
    logic [31:0] dot_product_accum [EMBEDDING_DIM/16]; // Accumulate in parallel
    logic [$clog2(EMBEDDING_DIM/16):0] accum_idx;
    
    // State machine logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= SC_IDLE;
            dot_product <= '0;
            norm_a <= '0;
            norm_b <= '0;
            distance_sq <= '0;
            epsilon <= EPSILON;
            accum_idx <= '0;
            done <= 1'b0;
            similarity <= '0;
            
            // Initialize accumulators
            for (int i = 0; i < EMBEDDING_DIM/16; i++) begin
                dot_product_accum[i] <= '0;
            end
        end else begin
            current_state <= next_state;
            
            case (current_state)
                SC_IDLE: begin
                    if (start) begin
                        dot_product <= '0;
                        norm_a <= '0;
                        norm_b <= '0;
                        distance_sq <= '0;
                        accum_idx <= '0;
                        done <= 1'b0;
                        
                        // Reset accumulators
                        for (int i = 0; i < EMBEDDING_DIM/16; i++) begin
                            dot_product_accum[i] <= '0;
                        end
                    end
                end
                
                SC_DOT_PRODUCT: begin
                    // Compute dot product using parallel accumulators
                    for (int i = 0; i < EMBEDDING_DIM/16; i++) begin
                        // Compute sub-dot products in parallel
                        dot_product_accum[i] <= '0;
                        for (int j = 0; j < 16; j++) begin
                            int idx = i*16 + j;
                            if (idx < EMBEDDING_DIM) begin
                                dot_product_accum[i] <= dot_product_accum[i] + (vec_a[idx] * vec_b[idx]);
                            end
                        end
                    end
                    
                    // Reduce accumulators to single dot product
                    if (accum_idx < EMBEDDING_DIM/16) begin
                        dot_product <= dot_product + dot_product_accum[accum_idx];
                        accum_idx <= accum_idx + 1;
                    end
                    
                    // For dot product similarity, we're done
                    if (metric == 2'b01 && accum_idx >= EMBEDDING_DIM/16) begin
                        similarity <= dot_product;
                    end
                end
                
                SC_COMPUTE_NORMS: begin
                    // Compute norms in parallel
                    if (metric == 2'b00) begin // Cosine similarity
                        // Compute squared norms
                        norm_a <= '0;
                        norm_b <= '0;
                        for (int i = 0; i < EMBEDDING_DIM; i++) begin
                            norm_a <= norm_a + (vec_a[i] * vec_a[i]);
                            norm_b <= norm_b + (vec_b[i] * vec_b[i]);
                        end
                    end else if (metric == 2'b10) begin // Euclidean distance
                        // Compute squared differences
                        distance_sq <= '0;
                        for (int i = 0; i < EMBEDDING_DIM; i++) begin
                            logic [31:0] diff = vec_a[i] - vec_b[i];
                            distance_sq <= distance_sq + (diff * diff);
                        end
                    end
                end
                
                SC_COSINE: begin
                    // Compute cosine similarity: dot_product / (|a| * |b|)
                    if (norm_a > 0 && norm_b > 0) begin
                        logic [31:0] norm_product = $sqrt(norm_a * norm_b + epsilon);
                        similarity <= dot_product / norm_product;
                    end else begin
                        similarity <= '0; // Handle zero vectors
                    end
                end
                
                SC_EUCLIDEAN: begin
                    // Compute Euclidean similarity: 1 / (1 + distance)
                    logic [31:0] distance = $sqrt(distance_sq);
                    similarity <= 32'h3f800000 / (32'h3f800000 + distance); // 1.0f / (1.0f + distance)
                end
                
                SC_FINISHED: begin
                    done <= 1'b1;
                end
            endcase
        end
    end
    
    // Next state logic with optimized decision path for each metric
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            SC_IDLE: begin
                if (start) begin
                    next_state = SC_DOT_PRODUCT;
                end
            end
            
            SC_DOT_PRODUCT: begin
                if (accum_idx >= EMBEDDING_DIM/16) begin
                    if (metric == 2'b01) begin // Dot product only
                        next_state = SC_FINISHED;
                    else if (metric == 2'b00) begin // Cosine similarity
                        next_state = SC_COMPUTE_NORMS;
                    end else begin // Euclidean distance
                        next_state = SC_COMPUTE_NORMS;
                    end
                end
            end
            
            SC_COMPUTE_NORMS: begin
                if (metric == 2'b00) begin // Cosine similarity
                    next_state = SC_COSINE;
                end else begin // Euclidean distance
                    next_state = SC_EUCLIDEAN;
                end
            end
            
            SC_COSINE: begin
                next_state = SC_FINISHED;
            end
            
            SC_EUCLIDEAN: begin
                next_state = SC_FINISHED;
            end
            
            SC_FINISHED: begin
                next_state = SC_IDLE;
            end
            
            default: next_state = SC_IDLE;
        endcase
    end

endmodule
