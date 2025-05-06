//-----------------------------------------------------------------------------
// File: qkv_projection.sv
// 
// Description: QKV Projection Module for RAG-CSD
//              Projects input embeddings to Query, Key, and Value vectors
//              for multi-head attention
//
// Parameters:
//   EMBEDDING_DIM  - Embedding dimension
//   HEAD_DIM       - Dimension per attention head
//   NUM_HEADS      - Number of attention heads
//   MAX_TOKENS     - Maximum tokens per query
//   BUS_WIDTH      - AXI bus width
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module qkv_projection #(
    parameter int EMBEDDING_DIM = 384,
    parameter int HEAD_DIM = 64,
    parameter int NUM_HEADS = 6,
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
    
    // Input embeddings
    input  logic [MAX_TOKENS-1:0][EMBEDDING_DIM-1:0][31:0]  input_embeddings,
    
    // Output projections
    output logic [NUM_HEADS-1:0][MAX_TOKENS-1:0][HEAD_DIM-1:0][31:0] q_proj,
    output logic [NUM_HEADS-1:0][MAX_TOKENS-1:0][HEAD_DIM-1:0][31:0] k_proj,
    output logic [NUM_HEADS-1:0][MAX_TOKENS-1:0][HEAD_DIM-1:0][31:0] v_proj,
    
    // Weight base address
    input  logic [31:0]                                   weight_base_addr,
    
    // Memory interface
    output logic                                          mem_rd_en,
    output logic [31:0]                                   mem_rd_addr,
    input  logic [BUS_WIDTH-1:0]                          mem_rd_data,
    input  logic                                          mem_rd_valid
);

    // QKV projection states
    typedef enum logic [3:0] {
        QKV_IDLE,
        QKV_LOAD_WEIGHTS,
        QKV_COMPUTE_Q,
        QKV_COMPUTE_K,
        QKV_COMPUTE_V,
        QKV_FINISHED
    } qkv_state_t;
    
    qkv_state_t current_state, next_state;
    
    // Internal registers
    logic [$clog2(MAX_TOKENS):0] token_idx;
    logic [$clog2(NUM_HEADS)-1:0] head_idx;
    logic [$clog2(HEAD_DIM):0] dim_idx;
    
    // Weight matrices
    logic [NUM_HEADS-1:0][HEAD_DIM-1:0][EMBEDDING_DIM-1:0][31:0] q_weights;
    logic [NUM_HEADS-1:0][HEAD_DIM-1:0][EMBEDDING_DIM-1:0][31:0] k_weights;
    logic [NUM_HEADS-1:0][HEAD_DIM-1:0][EMBEDDING_DIM-1:0][31:0] v_weights;
    
    // Memory interface state
    logic weights_loaded;
    logic [31:0] q_weights_addr, k_weights_addr, v_weights_addr;
    logic [31:0] weight_offset;
    
    // State machine logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= QKV_IDLE;
            token_idx <= '0;
            head_idx <= '0;
            dim_idx <= '0;
            weights_loaded <= 1'b0;
            mem_rd_en <= 1'b0;
            mem_rd_addr <= '0;
            weight_offset <= '0;
            done <= 1'b0;
            
            // Initialize weight addresses
            q_weights_addr <= '0;
            k_weights_addr <= '0;
            v_weights_addr <= '0;
        end else begin
            current_state <= next_state;
            
            case (current_state)
                QKV_IDLE: begin
                    if (start) begin
                        token_idx <= '0;
                        head_idx <= '0;
                        dim_idx <= '0;
                        weights_loaded <= 1'b0;
                        done <= 1'b0;
                        
                        // Calculate weight addresses
                        q_weights_addr <= weight_base_addr;
                        k_weights_addr <= weight_base_addr + (NUM_HEADS * HEAD_DIM * EMBEDDING_DIM * 4);
                        v_weights_addr <= weight_base_addr + (2 * NUM_HEADS * HEAD_DIM * EMBEDDING_DIM * 4);
                    end
                end
                
                QKV_LOAD_WEIGHTS: begin
                    // In a real implementation, we would load weights from memory here
                    // For simulation, we'll assume a simplified approach
                    
                    if (!weights_loaded) begin
                        // Simulate weight loading (in real hardware this would read from memory)
                        weights_loaded <= 1'b1;
                        
                        // In a real implementation, weights would be loaded from memory:
                        // mem_rd_en <= 1'b1;
                        // mem_rd_addr <= q_weights_addr + weight_offset;
                        // weight_offset <= weight_offset + BUS_WIDTH/8;
                    end
                end
                
                QKV_COMPUTE_Q: begin
                    if (token_idx < sequence_length) begin
                        // Compute Q projections for all heads in parallel
                        for (int h = 0; h < NUM_HEADS; h++) begin
                            for (int d = 0; d < HEAD_DIM; d++) begin
                                // Initialize accumulator
                                logic [31:0] q_val = 32'h0;
                                
                                // Matrix multiplication
                                for (int e = 0; e < EMBEDDING_DIM; e++) begin
                                    q_val = q_val + (input_embeddings[token_idx][e] * q_weights[h][d][e]);
                                end
                                
                                // Store result
                                q_proj[h][token_idx][d] <= q_val;
                            end
                        end
                        
                        token_idx <= token_idx + 1;
                    end
                end
                
                QKV_COMPUTE_K: begin
                    if (token_idx < sequence_length) begin
                        // Compute K projections for all heads in parallel
                        for (int h = 0; h < NUM_HEADS; h++) begin
                            for (int d = 0; d < HEAD_DIM; d++) begin
                                // Initialize accumulator
                                logic [31:0] k_val = 32'h0;
                                
                                // Matrix multiplication
                                for (int e = 0; e < EMBEDDING_DIM; e++) begin
                                    k_val = k_val + (input_embeddings[token_idx][e] * k_weights[h][d][e]);
                                end
                                
                                // Store result
                                k_proj[h][token_idx][d] <= k_val;
                            end
                        end
                        
                        token_idx <= token_idx + 1;
                    end
                end
                
                QKV_COMPUTE_V: begin
                    if (token_idx < sequence_length) begin
                        // Compute V projections for all heads in parallel
                        for (int h = 0; h < NUM_HEADS; h++) begin
                            for (int d = 0; d < HEAD_DIM; d++) begin
                                // Initialize accumulator
                                logic [31:0] v_val = 32'h0;
                                
                                // Matrix multiplication
                                for (int e = 0; e < EMBEDDING_DIM; e++) begin
                                    v_val = v_val + (input_embeddings[token_idx][e] * v_weights[h][d][e]);
                                end
                                
                                // Store result
                                v_proj[h][token_idx][d] <= v_val;
                            end
                        end
                        
                        token_idx <= token_idx + 1;
                    end
                end
                
                QKV_FINISHED: begin
                    done <= 1'b1;
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            QKV_IDLE: begin
                if (start) begin
                    next_state = QKV_LOAD_WEIGHTS;
                end
            end
            
            QKV_LOAD_WEIGHTS: begin
                if (weights_loaded) begin
                    next_state = QKV_COMPUTE_Q;
                end
            end
            
            QKV_COMPUTE_Q: begin
                if (token_idx >= sequence_length) begin
                    next_state = QKV_COMPUTE_K;
                    token_idx = '0;
                end
            end
            
            QKV_COMPUTE_K: begin
                if (token_idx >= sequence_length) begin
                    next_state = QKV_COMPUTE_V;
                    token_idx = '0;
                end
            end
            
            QKV_COMPUTE_V: begin
                if (token_idx >= sequence_length) begin
                    next_state = QKV_FINISHED;
                end
            end
            
            QKV_FINISHED: begin
                next_state = QKV_IDLE;
            end
            
            default: next_state = QKV_IDLE;
        endcase
    end

endmodule
