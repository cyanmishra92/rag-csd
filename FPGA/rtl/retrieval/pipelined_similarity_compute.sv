//-----------------------------------------------------------------------------
// File: pipelined_similarity_compute.sv
// 
// Description: Pipelined Similarity Compute Module for RAG-CSD
//              Computes similarity between vectors with deep pipeline
//              for higher throughput
//
// Parameters:
//   EMBEDDING_DIM      - Embedding dimension
//   PIPELINE_STAGES    - Number of pipeline stages
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module pipelined_similarity_compute #(
    parameter int EMBEDDING_DIM = 384,
    parameter int PIPELINE_STAGES = 4
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
    output logic [31:0]                     similarity,
    
    // Pipeline status
    output logic                            pipeline_full,
    output logic                            pipeline_empty
);

    // Define how many elements to process per stage
    localparam int ELEMENTS_PER_STAGE = EMBEDDING_DIM / PIPELINE_STAGES;
    
    // Pipeline registers
    logic [PIPELINE_STAGES-1:0] stage_valid;
    logic [1:0] stage_metric [PIPELINE_STAGES-1:0];
    logic [31:0] stage_dot_product [PIPELINE_STAGES-1:0];
    logic [31:0] stage_norm_a [PIPELINE_STAGES-1:0];
    logic [31:0] stage_norm_b [PIPELINE_STAGES-1:0];
    logic [31:0] stage_distance_sq [PIPELINE_STAGES-1:0];
    
    // Final computation registers
    logic [31:0] final_dot_product;
    logic [31:0] final_norm_a;
    logic [31:0] final_norm_b;
    logic [31:0] final_distance_sq;
    logic [1:0] final_metric;
    logic final_valid;
    
    // Constants
    localparam logic [31:0] EPSILON = 32'h3a83126f; // 0.001f in IEEE-754
    
    // Pipeline status flags
    assign pipeline_full = &stage_valid;
    assign pipeline_empty = ~|stage_valid;
    
    // Pipeline initialization and control
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset pipeline
            stage_valid <= '0;
            final_valid <= 1'b0;
            done <= 1'b0;
            
            for (int i = 0; i < PIPELINE_STAGES; i++) begin
                stage_metric[i] <= '0;
                stage_dot_product[i] <= '0;
                stage_norm_a[i] <= '0;
                stage_norm_b[i] <= '0;
                stage_distance_sq[i] <= '0;
            end
            
            final_dot_product <= '0;
            final_norm_a <= '0;
            final_norm_b <= '0;
            final_distance_sq <= '0;
            final_metric <= '0;
        end else begin
            // Default pulse signals
            done <= 1'b0;
            
            // First stage: accept new input if start is asserted
            if (start) begin
                stage_valid[0] <= 1'b1;
                stage_metric[0] <= metric;
                stage_dot_product[0] <= '0;
                stage_norm_a[0] <= '0;
                stage_norm_b[0] <= '0;
                stage_distance_sq[0] <= '0;
                
                // Process first chunk of elements
                for (int i = 0; i < ELEMENTS_PER_STAGE; i++) begin
                    // Dot product calculation
                    stage_dot_product[0] <= stage_dot_product[0] + (vec_a[i] * vec_b[i]);
                    
                    // Norm calculations for cosine similarity
                    stage_norm_a[0] <= stage_norm_a[0] + (vec_a[i] * vec_a[i]);
                    stage_norm_b[0] <= stage_norm_b[0] + (vec_b[i] * vec_b[i]);
                    
                    // Distance calculation for Euclidean
                    logic [31:0] diff = vec_a[i] - vec_b[i];
                    stage_distance_sq[0] <= stage_distance_sq[0] + (diff * diff);
                end
            end
            
            // Middle stages: process each chunk of elements
            for (int s = 1; s < PIPELINE_STAGES; s++) begin
                if (stage_valid[s-1]) begin
                    // Pass control signals
                    stage_valid[s] <= stage_valid[s-1];
                    stage_metric[s] <= stage_metric[s-1];
                    
                    // Pass accumulated values
                    stage_dot_product[s] <= stage_dot_product[s-1];
                    stage_norm_a[s] <= stage_norm_a[s-1];
                    stage_norm_b[s] <= stage_norm_b[s-1];
                    stage_distance_sq[s] <= stage_distance_sq[s-1];
                    
                    // Process this chunk of elements
                    for (int i = s*ELEMENTS_PER_STAGE; 
                         i < (s+1)*ELEMENTS_PER_STAGE && i < EMBEDDING_DIM; 
                         i++) begin
                        // Dot product calculation
                        stage_dot_product[s] <= stage_dot_product[s] + (vec_a[i] * vec_b[i]);
                        
                        // Norm calculations for cosine similarity
                        stage_norm_a[s] <= stage_norm_a[s] + (vec_a[i] * vec_a[i]);
                        stage_norm_b[s] <= stage_norm_b[s] + (vec_b[i] * vec_b[i]);
                        
                        // Distance calculation for Euclidean
                        logic [31:0] diff = vec_a[i] - vec_b[i];
                        stage_distance_sq[s] <= stage_distance_sq[s] + (diff * diff);
                    end
                end else begin
                    stage_valid[s] <= 1'b0;
                end
            end
            
            // Final stage: compute final result
            if (stage_valid[PIPELINE_STAGES-1]) begin
                final_valid <= 1'b1;
                final_dot_product <= stage_dot_product[PIPELINE_STAGES-1];
                final_norm_a <= stage_norm_a[PIPELINE_STAGES-1];
                final_norm_b <= stage_norm_b[PIPELINE_STAGES-1];
                final_distance_sq <= stage_distance_sq[PIPELINE_STAGES-1];
                final_metric <= stage_metric[PIPELINE_STAGES-1];
            end else begin
                final_valid <= 1'b0;
            end
            
            // Compute final similarity based on metric
            if (final_valid) begin
                case (final_metric)
                    2'b00: begin // Cosine similarity
                        logic [31:0] norm_product = $sqrt(final_norm_a * final_norm_b + EPSILON);
                        similarity <= final_dot_product / norm_product;
                    end
                    2'b01: begin // Dot product
                        similarity <= final_dot_product;
                    end
                    2'b10: begin // Euclidean distance
                        logic [31:0] distance = $sqrt(final_distance_sq);
                        similarity <= 32'h3f800000 / (32'h3f800000 + distance); // 1.0f / (1.0f + distance)
                    end
                    default: begin
                        similarity <= 32'h0; // Default to zero
                    end
                endcase
                
                done <= 1'b1;
            end
        end
    end

endmodule
