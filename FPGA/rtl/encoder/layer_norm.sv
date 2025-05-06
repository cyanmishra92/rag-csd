//-----------------------------------------------------------------------------
// File: layer_norm.sv
// 
// Description: Layer Normalization Module for RAG-CSD
//              Implements layer normalization for transformer
//
// Parameters:
//   EMBEDDING_DIM  - Embedding dimension
//   MAX_TOKENS     - Maximum tokens per query
//   BUS_WIDTH      - AXI bus width
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module layer_norm #(
    parameter int EMBEDDING_DIM = 384,
    parameter int MAX_TOKENS = 128,
    parameter int BUS_WIDTH = 512
) (
    // Clock and reset
    input  logic                                          clk,
    input  logic                                          rst_n,
    
    // Control signals
    input  logic                                          start,
    output logic                                          done,
    
    // Sequence info
    input  logic [$clog2(MAX_TOKENS):0]                   sequence_length,
    
    // Input and output embeddings
    input  logic [MAX_TOKENS-1:0][EMBEDDING_DIM-1:0][31:0]  input_embeddings,
    output logic [MAX_TOKENS-1:0][EMBEDDING_DIM-1:0][31:0]  output_embeddings,
    
    // Weight base address
    input  logic [31:0]                                   weight_base_addr,
    
    // Memory interface
    output logic                                          mem_rd_en,
    output logic [31:0]                                   mem_rd_addr,
    input  logic [BUS_WIDTH-1:0]                          mem_rd_data,
    input  logic                                          mem_rd_valid
);

    // Layer normalization states
    typedef enum logic [3:0] {
        LN_IDLE,
        LN_LOAD_PARAMS,
        LN_COMPUTE_MEAN,
        LN_COMPUTE_VAR,
        LN_NORMALIZE,
        LN_SCALE_SHIFT,
        LN_FINISHED
    } ln_state_t;
    
    ln_state_t current_state, next_state;
    
    // Internal registers
    logic [$clog2(MAX_TOKENS):0] token_idx;
    logic [$clog2(EMBEDDING_DIM):0] dim_idx;
    logic [31:0] mean [MAX_TOKENS-1:0];
    logic [31:0] variance [MAX_TOKENS-1:0];
    logic [31:0] epsilon;
    
    // Normalization parameters (gamma, beta)
    logic [EMBEDDING_DIM-1:0][31:0] gamma;
    logic [EMBEDDING_DIM-1:0][31:0] beta;
    
    // Memory interface state
    logic weights_loaded;
    logic [31:0] gamma_addr, beta_addr;
    logic [31:0] mem_words_to_read, mem_words_read;
    
    // Constants
    localparam logic [31:0] EPSILON = 32'h3a83126f; // 0.001f in IEEE-754
    
    // State machine logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= LN_IDLE;
            token_idx <= '0;
            dim_idx <= '0;
            weights_loaded <= 1'b0;
            mem_rd_en <= 1'b0;
            mem_rd_addr <= '0;
            mem_words_to_read <= '0;
            mem_words_read <= '0;
            epsilon <= EPSILON;
            done <= 1'b0;
            
            // Initialize means and variances
            for (int i = 0; i < MAX_TOKENS; i++) begin
                mean[i] <= '0;
                variance[i] <= '0;
            end
            
            // Initialize gamma and beta
            for (int i = 0; i < EMBEDDING_DIM; i++) begin
                gamma[i] <= 32'h3f800000; // 1.0f in IEEE-754
                beta[i] <= 32'h00000000;  // 0.0f in IEEE-754
            end
        end else begin
            current_state <= next_state;
            
            case (current_state)
                LN_IDLE: begin
                    if (start) begin
                        token_idx <= '0;
                        dim_idx <= '0;
                        weights_loaded <= 1'b0;
                        done <= 1'b0;
                        
                        // Calculate parameter addresses
                        gamma_addr <= weight_base_addr;
                        beta_addr <= weight_base_addr + (EMBEDDING_DIM * 4);
                    end
                end
                
                LN_LOAD_PARAMS: begin
                    // Load gamma and beta parameters
                    if (!weights_loaded) begin
                        if (dim_idx < EMBEDDING_DIM) begin
                            // Read gamma
                            mem_rd_en <= 1'b1;
                            mem_rd_addr <= gamma_addr + (dim_idx * 4);
                            dim_idx <= dim_idx + 1;
                        end else if (dim_idx < EMBEDDING_DIM * 2) begin
                            // Read beta
                            mem_rd_en <= 1'b1;
                            mem_rd_addr <= beta_addr + ((dim_idx - EMBEDDING_DIM) * 4);
                            dim_idx <= dim_idx + 1;
                        end else begin
                            mem_rd_en <= 1'b0;
                            weights_loaded <= 1'b1;
                        end
                        
                        // Process memory read data
                        if (mem_rd_valid) begin
                            if (dim_idx <= EMBEDDING_DIM) begin
                                gamma[dim_idx-1] <= mem_rd_data[31:0];
                            end else begin
                                beta[dim_idx-EMBEDDING_DIM-1] <= mem_rd_data[31:0];
                            end
                        end
                    end
                end
                
                LN_COMPUTE_MEAN: begin
                    if (token_idx < sequence_length) begin
                        // Compute mean for current token
                        logic [31:0] sum = '0;
                        for (int i = 0; i < EMBEDDING_DIM; i++) begin
                            sum = sum + input_embeddings[token_idx][i];
                        end
                        mean[token_idx] <= sum / EMBEDDING_DIM;
                        token_idx <= token_idx + 1;
                    end
                end
                
                LN_COMPUTE_VAR: begin
                    if (token_idx < sequence_length) begin
                        // Compute variance for current token
                        logic [31:0] sum_sq_diff = '0;
                        for (int i = 0; i < EMBEDDING_DIM; i++) begin
                            logic [31:0] diff = input_embeddings[token_idx][i] - mean[token_idx];
                            sum_sq_diff = sum_sq_diff + (diff * diff);
                        end
                        variance[token_idx] <= sum_sq_diff / EMBEDDING_DIM;
                        token_idx <= token_idx + 1;
                    end
                end
                
                LN_NORMALIZE: begin
                    if (token_idx < sequence_length) begin
                        // Normalize current token
                        for (int i = 0; i < EMBEDDING_DIM; i++) begin
                            logic [31:0] diff = input_embeddings[token_idx][i] - mean[token_idx];
                            logic [31:0] std_dev = $sqrt(variance[token_idx] + epsilon);
                            logic [31:0] normalized = diff / std_dev;
                            output_embeddings[token_idx][i] <= normalized;
                        end
                        token_idx <= token_idx + 1;
                    end
                end
                
                LN_SCALE_SHIFT: begin
                    if (token_idx < sequence_length) begin
                        // Apply scale and shift (gamma and beta)
                        for (int i = 0; i < EMBEDDING_DIM; i++) begin
                            output_embeddings[token_idx][i] <= 
                                (output_embeddings[token_idx][i] * gamma[i]) + beta[i];
                        end
                        token_idx <= token_idx + 1;
                    end
                end
                
                LN_FINISHED: begin
                    done <= 1'b1;
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            LN_IDLE: begin
                if (start) begin
                    next_state = LN_LOAD_PARAMS;
                end
            end
            
            LN_LOAD_PARAMS: begin
                if (weights_loaded) begin
                    next_state = LN_COMPUTE_MEAN;
                end
            end
            
            LN_COMPUTE_MEAN: begin
                if (token_idx >= sequence_length) begin
                    next_state = LN_COMPUTE_VAR;
                    token_idx = '0;
                end
            end
            
            LN_COMPUTE_VAR: begin
                if (token_idx >= sequence_length) begin
                    next_state = LN_NORMALIZE;
                    token_idx = '0;
                end
            end
            
            LN_NORMALIZE: begin
                if (token_idx >= sequence_length) begin
                    next_state = LN_SCALE_SHIFT;
                    token_idx = '0;
                end
            end
            
            LN_SCALE_SHIFT: begin
                if (token_idx >= sequence_length) begin
                    next_state = LN_FINISHED;
                end
            end
            
            LN_FINISHED: begin
                next_state = LN_IDLE;
            end
            
            default: next_state = LN_IDLE;
        endcase
    end

endmodule
