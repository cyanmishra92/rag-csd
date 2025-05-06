//-----------------------------------------------------------------------------
// File: self_attention.sv
// 
// Description: Self-Attention Module for RAG-CSD
//              Implements multi-head self-attention mechanism
//
// Parameters:
//   HEAD_DIM       - Dimension per attention head
//   NUM_HEADS      - Number of attention heads
//   MAX_TOKENS     - Maximum tokens per query
//   EMBEDDING_DIM  - Total embedding dimension
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module self_attention #(
    parameter int HEAD_DIM = 64,
    parameter int NUM_HEADS = 6,
    parameter int MAX_TOKENS = 128,
    parameter int EMBEDDING_DIM = 384
) (
    // Clock and reset
    input  logic                                          clk,
    input  logic                                          rst_n,
    
    // Control signals
    input  logic                                          start,
    output logic                                          done,
    
    // Sequence info
    input  logic [$clog2(MAX_TOKENS):0]                   sequence_length,
    
    // Input Q, K, V projections
    input  logic [NUM_HEADS-1:0][MAX_TOKENS-1:0][HEAD_DIM-1:0][31:0] q_proj,
    input  logic [NUM_HEADS-1:0][MAX_TOKENS-1:0][HEAD_DIM-1:0][31:0] k_proj,
    input  logic [NUM_HEADS-1:0][MAX_TOKENS-1:0][HEAD_DIM-1:0][31:0] v_proj,
    
    // Output projection
    output logic [MAX_TOKENS-1:0][EMBEDDING_DIM-1:0][31:0] output_projection,
    
    // Weight base address for output projection
    input  logic [31:0]                                   weight_base_addr,
    
    // Memory interface
    output logic                                          mem_rd_en,
    output logic [31:0]                                   mem_rd_addr,
    input  logic [BUS_WIDTH-1:0]                          mem_rd_data,
    input  logic                                          mem_rd_valid
);

    // Parameter declarations
    parameter int BUS_WIDTH = 512;
    
    // Self-attention states
    typedef enum logic [3:0] {
        SA_IDLE,
        SA_LOAD_WEIGHTS,
        SA_COMPUTE_QK,
        SA_SOFTMAX,
        SA_COMPUTE_ATTENTION,
        SA_OUTPUT_PROJECTION,
        SA_FINISHED
    } sa_state_t;
    
    sa_state_t current_state, next_state;
    
    // Internal registers
    logic [$clog2(NUM_HEADS)-1:0] head_idx;
    logic [$clog2(MAX_TOKENS):0] token_idx, context_idx;
    
    // Attention scores and weights
    logic [NUM_HEADS-1:0][MAX_TOKENS-1:0][MAX_TOKENS-1:0][31:0] attn_scores;
    logic [NUM_HEADS-1:0][MAX_TOKENS-1:0][MAX_TOKENS-1:0][31:0] attn_weights;
    
    // Attention outputs
    logic [NUM_HEADS-1:0][MAX_TOKENS-1:0][HEAD_DIM-1:0][31:0] attn_output;
    logic [MAX_TOKENS-1:0][EMBEDDING_DIM-1:0][31:0] concatenated_output;
    
    // Output projection weights
    logic [EMBEDDING_DIM-1:0][EMBEDDING_DIM-1:0][31:0] output_weights;
    
    // Memory interface state
    logic weights_loaded;
    logic [31:0] weight_offset;
    
    // Constants
    localparam logic [31:0] SCALE_FACTOR = 32'h3F000000; // 0.5 in IEEE-754 (approximation of 1/sqrt(HEAD_DIM))
    
    // State machine logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= SA_IDLE;
            head_idx <= '0;
            token_idx <= '0;
            context_idx <= '0;
            weights_loaded <= 1'b0;
            mem_rd_en <= 1'b0;
            mem_rd_addr <= '0;
            weight_offset <= '0;
            done <= 1'b0;
        end else begin
            current_state <= next_state;
            
            case (current_state)
                SA_IDLE: begin
                    if (start) begin
                        head_idx <= '0;
                        token_idx <= '0;
                        context_idx <= '0;
                        weights_loaded <= 1'b0;
                        done <= 1'b0;
                    end
                end
                
                SA_LOAD_WEIGHTS: begin
                    // In a real implementation, we would load output projection weights
                    // For simulation, we'll assume they're accessible directly
                    weights_loaded <= 1'b1;
                end
                
                SA_COMPUTE_QK: begin
                    // Process one head at a time for simplicity
                    if (head_idx < NUM_HEADS) begin
                        // Process one query token at a time
                        if (token_idx < sequence_length) begin
                            // Compute attention scores for this token against all context tokens
                            if (context_idx < sequence_length) begin
                                // Compute Q·K^T (dot product of query and key)
                                logic [31:0] dot_product = 32'h0;
                                
                                for (int d = 0; d < HEAD_DIM; d++) begin
                                    dot_product = dot_product + 
                                        (q_proj[head_idx][token_idx][d] * k_proj[head_idx][context_idx][d]);
                                end
                                
                                // Scale dot product by 1/sqrt(d_k)
                                attn_scores[head_idx][token_idx][context_idx] <= dot_product * SCALE_FACTOR;
                                
                                context_idx <= context_idx + 1;
                            end else begin
                                // Move to next token
                                token_idx <= token_idx + 1;
                                context_idx <= '0;
                            end
                        end else begin
                            // Move to next head
                            head_idx <= head_idx + 1;
                            token_idx <= '0;
                            context_idx <= '0;
                        end
                    end
                end
                
                SA_SOFTMAX: begin
                    // Process one head at a time
                    if (head_idx < NUM_HEADS) begin
                        // Process one query token at a time
                        if (token_idx < sequence_length) begin
                            // Compute softmax for this token's attention scores
                            
                            // Step 1: Find maximum score for numerical stability
                            logic [31:0] max_score = 32'h80000000; // Minimum possible float value
                            for (int i = 0; i < sequence_length; i++) begin
                                if (attn_scores[head_idx][token_idx][i] > max_score) begin
                                    max_score = attn_scores[head_idx][token_idx][i];
                                end
                            end
                            
                            // Step 2: Compute exp(score - max_score) for each score
                            logic [31:0] exp_sum = 32'h0;
                            for (int i = 0; i < sequence_length; i++) begin
                                // In real hardware, this would use a lookup table or approximation
                                logic [31:0] shifted_score = attn_scores[head_idx][token_idx][i] - max_score;
                                logic [31:0] exp_score = $exp(shifted_score); // Simulation only
                                attn_weights[head_idx][token_idx][i] = exp_score;
                                exp_sum = exp_sum + exp_score;
                            end
                            
                            // Step 3: Normalize by dividing by the sum
                            for (int i = 0; i < sequence_length; i++) begin
                                attn_weights[head_idx][token_idx][i] = 
                                    attn_weights[head_idx][token_idx][i] / exp_sum;
                            end
                            
                            token_idx <= token_idx + 1;
                        end else begin
                            head_idx <= head_idx + 1;
                            token_idx <= '0;
                        end
                    end
                end
                
                SA_COMPUTE_ATTENTION: begin
                    // Process one head at a time
                    if (head_idx < NUM_HEADS) begin
                        // Process one query token at a time
                        if (token_idx < sequence_length) begin
                            // Compute weighted sum of values for each dimension
                            for (int d = 0; d < HEAD_DIM; d++) begin
                                logic [31:0] weighted_sum = 32'h0;
                                
                                for (int i = 0; i < sequence_length; i++) begin
                                    weighted_sum = weighted_sum + 
                                        (attn_weights[head_idx][token_idx][i] * v_proj[head_idx][i][d]);
                                end
                                
                                attn_output[head_idx][token_idx][d] <= weighted_sum;
                            end
                            
                            token_idx <= token_idx + 1;
                        end else begin
                            head_idx <= head_idx + 1;
                            token_idx <= '0;
                        end
                    end
                end
                
                SA_OUTPUT_PROJECTION: begin
                    // Concatenate all attention heads and project to output dimension
                    if (token_idx < sequence_length) begin
                        // First, concatenate all attention heads for this token
                        for (int h = 0; h < NUM_HEADS; h++) begin
                            for (int d = 0; d < HEAD_DIM; d++) begin
                                concatenated_output[token_idx][h*HEAD_DIM+d] <= attn_output[h][token_idx][d];
                            end
                        end
                        
                        // Then compute output projection
                        for (int o = 0; o < EMBEDDING_DIM; o++) begin
                            logic [31:0] proj_sum = 32'h0;
                            
                            for (int i = 0; i < EMBEDDING_DIM; i++) begin
                                proj_sum = proj_sum + 
                                    (concatenated_output[token_idx][i] * output_weights[o][i]);
                            end
                            
                            output_projection[token_idx][o] <= proj_sum;
                        end
                        
                        token_idx <= token_idx + 1;
                    end
                end
                
                SA_FINISHED: begin
                    done <= 1'b1;
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            SA_IDLE: begin
                if (start) begin
                    next_state = SA_LOAD_WEIGHTS;
                end
            end
            
            SA_LOAD_WEIGHTS: begin
                if (weights_loaded) begin
                    next_state = SA_COMPUTE_QK;
                end
            end
            
            SA_COMPUTE_QK: begin
                if (head_idx >= NUM_HEADS) begin
                    next_state = SA_SOFTMAX;
                    head_idx = '0;
                    token_idx = '0;
                end
            end
            
            SA_SOFTMAX: begin
                if (head_idx >= NUM_HEADS) begin
                    next_state = SA_COMPUTE_ATTENTION;
                    head_idx = '0;
                    token_idx = '0;
                end
            end
            
            SA_COMPUTE_ATTENTION: begin
                if (head_idx >= NUM_HEADS) begin
                    next_state = SA_OUTPUT_PROJECTION;
                    token_idx = '0;
                end
            end
            
            SA_OUTPUT_PROJECTION: begin
                if (token_idx >= sequence_length) begin
                    next_state = SA_FINISHED;
                end
            end
            
            SA_FINISHED: begin
                next_state = SA_IDLE;
            end
            
            default: next_state = SA_IDLE;
        endcase
    end

endmodule
