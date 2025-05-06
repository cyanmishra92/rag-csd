//-----------------------------------------------------------------------------
// File: feed_forward.sv
// 
// Description: Feed-Forward Module for RAG-CSD
//              Implements a feed-forward neural network layer with ReLU activation
//
// Parameters:
//   INPUT_DIM    - Input dimension
//   OUTPUT_DIM   - Output dimension
//   MAX_TOKENS   - Maximum tokens per query
//   BUS_WIDTH    - AXI bus width
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module feed_forward #(
    parameter int INPUT_DIM = 384,
    parameter int OUTPUT_DIM = 1536,
    parameter int MAX_TOKENS = 128,
    parameter int BUS_WIDTH = 512
) (
    // Clock and reset
    input  logic                                        clk,
    input  logic                                        rst_n,
    
    // Control signals
    input  logic                                        start,
    output logic                                        done,
    
    // Sequence info
    input  logic [$clog2(MAX_TOKENS):0]                 sequence_length,
    
    // Input and output embeddings
    input  logic [MAX_TOKENS-1:0][INPUT_DIM-1:0][31:0]  input_embeddings,
    output logic [MAX_TOKENS-1:0][OUTPUT_DIM-1:0][31:0] output_embeddings,
    
    // Weight base address
    input  logic [31:0]                                 weight_base_addr,
    
    // Memory interface
    output logic                                        mem_rd_en,
    output logic [31:0]                                 mem_rd_addr,
    input  logic [BUS_WIDTH-1:0]                        mem_rd_data,
    input  logic                                        mem_rd_valid
);

    // Feed-forward states
    typedef enum logic [3:0] {
        FF_IDLE,
        FF_LOAD_WEIGHTS,
        FF_COMPUTE_LINEAR,
        FF_APPLY_ACTIVATION,
        FF_FINISHED
    } ff_state_t;
    
    ff_state_t current_state, next_state;
    
    // Internal registers
    logic [$clog2(MAX_TOKENS):0] token_idx;
    logic [$clog2(OUTPUT_DIM):0] output_idx;
    
    // Weight matrix and bias
    logic [OUTPUT_DIM-1:0][INPUT_DIM-1:0][31:0] weights;
    logic [OUTPUT_DIM-1:0][31:0] biases;
    
    // Memory interface state
    logic weights_loaded;
    logic [31:0] weight_offset;
    
    // State machine logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= FF_IDLE;
            token_idx <= '0;
            output_idx <= '0;
            weights_loaded <= 1'b0;
            mem_rd_en <= 1'b0;
            mem_rd_addr <= '0;
            weight_offset <= '0;
            done <= 1'b0;
        end else begin
            current_state <= next_state;
            
            case (current_state)
                FF_IDLE: begin
                    if (start) begin
                        token_idx <= '0;
                        output_idx <= '0;
                        weights_loaded <= 1'b0;
                        done <= 1'b0;
                    end
                end
                
                FF_LOAD_WEIGHTS: begin
                    // In a real implementation, we would load weights from memory
                    // For simulation, we'll assume they're accessible directly
                    weights_loaded <= 1'b1;
                end
                
                FF_COMPUTE_LINEAR: begin
                    if (token_idx < sequence_length) begin
                        // Compute output_idx rows at a time for better efficiency
                        if (output_idx < OUTPUT_DIM) begin
                            // Linear transformation: output = weights * input + bias
                            logic [31:0] sum = biases[output_idx];
                            
                            for (int i = 0; i < INPUT_DIM; i++) begin
                                sum = sum + (weights[output_idx][i] * input_embeddings[token_idx][i]);
                            end
                            
                            output_embeddings[token_idx][output_idx] <= sum;
                            output_idx <= output_idx + 1;
                        end else begin
                            // Move to next token
                            token_idx <= token_idx + 1;
                            output_idx <= '0;
                        end
                    end
                end
                
                FF_APPLY_ACTIVATION: begin
                    if (token_idx < sequence_length) begin
                        // Apply ReLU activation: max(0, x)
                        for (int o = 0; o < OUTPUT_DIM; o++) begin
                            if (output_embeddings[token_idx][o] < 0) begin
                                output_embeddings[token_idx][o] <= 0;
                            end
                        end
                        
                        token_idx <= token_idx + 1;
                    end
                end
                
                FF_FINISHED: begin
                    done <= 1'b1;
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            FF_IDLE: begin
                if (start) begin
                    next_state = FF_LOAD_WEIGHTS;
                end
            end
            
            FF_LOAD_WEIGHTS: begin
                if (weights_loaded) begin
                    next_state = FF_COMPUTE_LINEAR;
                end
            end
            
            FF_COMPUTE_LINEAR: begin
                if (token_idx >= sequence_length) begin
                    next_state = FF_APPLY_ACTIVATION;
                    token_idx = '0;
                end
            end
            
            FF_APPLY_ACTIVATION: begin
                if (token_idx >= sequence_length) begin
                    next_state = FF_FINISHED;
                end
            end
            
            FF_FINISHED: begin
                next_state = FF_IDLE;
            end
            
            default: next_state = FF_IDLE;
        endcase
    end

endmodule
