//-----------------------------------------------------------------------------
// File: transformer_layer.sv
// 
// Description: Transformer Layer Module for RAG-CSD
//              Implements a single layer of transformer with self-attention 
//              and feed-forward networks
//
// Parameters:
//   EMBEDDING_DIM  - Embedding dimension
//   MAX_TOKENS     - Maximum tokens per query
//   HEAD_DIM       - Dimension per attention head
//   NUM_HEADS      - Number of attention heads
//   HIDDEN_DIM     - Hidden dimension in feed-forward
//   BUS_WIDTH      - AXI bus width
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module transformer_layer #(
    parameter int EMBEDDING_DIM = 384,
    parameter int MAX_TOKENS = 128,
    parameter int HEAD_DIM = 64,
    parameter int NUM_HEADS = 6,
    parameter int HIDDEN_DIM = 1536,
    parameter int BUS_WIDTH = 512
) (
    // Clock and reset
    input  logic                                          clk,
    input  logic                                          rst_n,
    
    // Control signals
    input  logic                                          start,
    output logic                                          done,
    input  logic [$clog2(6)-1:0]                          layer_idx,
    
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

    // Transformer layer state machine
    typedef enum logic [4:0] {
        TL_IDLE,
        TL_LOAD_WEIGHTS,
        TL_LAYER_NORM1,
        TL_QKV_PROJECT,
        TL_ATTENTION,
        TL_PROJECT_OUTPUT,
        TL_ADD_RESIDUAL1,
        TL_LAYER_NORM2,
        TL_FF1,
        TL_FF2,
        TL_ADD_RESIDUAL2,
        TL_FINISHED
    } transformer_layer_state_t;
    
    transformer_layer_state_t current_state, next_state;
    
    // Internal registers and signals
    logic weights_loaded;                  // Flag for weight loading
    logic [31:0] qkv_weights_addr;         // Address of QKV projection weights
    logic [31:0] out_weights_addr;         // Address of output projection weights
    logic [31:0] ff1_weights_addr;         // Address of feed-forward 1 weights
    logic [31:0] ff2_weights_addr;         // Address of feed-forward 2 weights
    
    // Component signals
    logic layernorm1_start, layernorm1_done;
    logic layernorm2_start, layernorm2_done;
    logic qkv_project_start, qkv_project_done;
    logic attention_start, attention_done;
    logic out_project_start, out_project_done;
    logic ff1_start, ff1_done;
    logic ff2_start, ff2_done;
    
    // Intermediate results
    logic [MAX_TOKENS-1:0][EMBEDDING_DIM-1:0][31:0] norm1_output;
    logic [MAX_TOKENS-1:0][EMBEDDING_DIM-1:0][31:0] attention_output;
    logic [MAX_TOKENS-1:0][EMBEDDING_DIM-1:0][31:0] norm2_output;
    logic [MAX_TOKENS-1:0][HIDDEN_DIM-1:0][31:0] ff1_output;
    
    // QKV projections
    logic [NUM_HEADS-1:0][MAX_TOKENS-1:0][HEAD_DIM-1:0][31:0] q_proj;
    logic [NUM_HEADS-1:0][MAX_TOKENS-1:0][HEAD_DIM-1:0][31:0] k_proj;
    logic [NUM_HEADS-1:0][MAX_TOKENS-1:0][HEAD_DIM-1:0][31:0] v_proj;
    
    // Memory interface state
    logic [31:0] mem_offset;               // Memory address offset
    logic [31:0] mem_words_to_read;        // Words to read from memory
    logic [31:0] mem_words_read;           // Words read from memory
    
    // Memory arbitration signals
    logic ln1_mem_rd_en, qkv_mem_rd_en, attn_mem_rd_en, out_mem_rd_en;
    logic ln2_mem_rd_en, ff1_mem_rd_en, ff2_mem_rd_en;
    logic [31:0] ln1_mem_addr, qkv_mem_addr, attn_mem_addr, out_mem_addr;
    logic [31:0] ln2_mem_addr, ff1_mem_addr, ff2_mem_addr;
    
    // Layer normalization 1
    layer_norm #(
        .EMBEDDING_DIM(EMBEDDING_DIM),
        .MAX_TOKENS(MAX_TOKENS),
        .BUS_WIDTH(BUS_WIDTH)
    ) layer_norm1_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(layernorm1_start),
        .done(layernorm1_done),
        .sequence_length(sequence_length),
        .input_embeddings(input_embeddings),
        .output_embeddings(norm1_output),
        .weight_base_addr(weight_base_addr),
        .mem_rd_en(ln1_mem_rd_en),
        .mem_rd_addr(ln1_mem_addr),
        .mem_rd_data(mem_rd_data),
        .mem_rd_valid(mem_rd_valid)
    );
    
    // QKV projection
    qkv_projection #(
        .EMBEDDING_DIM(EMBEDDING_DIM),
        .HEAD_DIM(HEAD_DIM),
        .NUM_HEADS(NUM_HEADS),
        .MAX_TOKENS(MAX_TOKENS),
        .BUS_WIDTH(BUS_WIDTH)
    ) qkv_proj_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(qkv_project_start),
        .done(qkv_project_done),
        .sequence_length(sequence_length),
        .input_embeddings(norm1_output),
        .q_proj(q_proj),
        .k_proj(k_proj),
        .v_proj(v_proj),
        .weight_base_addr(qkv_weights_addr),
        .mem_rd_en(qkv_mem_rd_en),
        .mem_rd_addr(qkv_mem_addr),
        .mem_rd_data(mem_rd_data),
        .mem_rd_valid(mem_rd_valid)
    );
    
    // Self-attention
    self_attention #(
        .HEAD_DIM(HEAD_DIM),
        .NUM_HEADS(NUM_HEADS),
        .MAX_TOKENS(MAX_TOKENS),
        .EMBEDDING_DIM(EMBEDDING_DIM)
    ) attention_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(attention_start),
        .done(attention_done),
        .sequence_length(sequence_length),
        .q_proj(q_proj),
        .k_proj(k_proj),
        .v_proj(v_proj),
        .output_projection(attention_output),
        .weight_base_addr(out_weights_addr),
        .mem_rd_en(attn_mem_rd_en),
        .mem_rd_addr(attn_mem_addr),
        .mem_rd_data(mem_rd_data),
        .mem_rd_valid(mem_rd_valid)
    );
    
    // Layer normalization 2
    layer_norm #(
        .EMBEDDING_DIM(EMBEDDING_DIM),
        .MAX_TOKENS(MAX_TOKENS),
        .BUS_WIDTH(BUS_WIDTH)
    ) layer_norm2_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(layernorm2_start),
        .done(layernorm2_done),
        .sequence_length(sequence_length),
        .input_embeddings(attention_output),
        .output_embeddings(norm2_output),
        .weight_base_addr(weight_base_addr + 32'h1000),
        .mem_rd_en(ln2_mem_rd_en),
        .mem_rd_addr(ln2_mem_addr),
        .mem_rd_data(mem_rd_data),
        .mem_rd_valid(mem_rd_valid)
    );
    
    // Feed-forward network first layer
    feed_forward #(
        .INPUT_DIM(EMBEDDING_DIM),
        .OUTPUT_DIM(HIDDEN_DIM),
        .MAX_TOKENS(MAX_TOKENS),
        .BUS_WIDTH(BUS_WIDTH)
    ) ff1_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(ff1_start),
        .done(ff1_done),
        .sequence_length(sequence_length),
        .input_embeddings(norm2_output),
        .output_embeddings(ff1_output),
        .weight_base_addr(ff1_weights_addr),
        .mem_rd_en(ff1_mem_rd_en),
        .mem_rd_addr(ff1_mem_addr),
        .mem_rd_data(mem_rd_data),
        .mem_rd_valid(mem_rd_valid)
    );
    
    // Feed-forward network second layer
    feed_forward #(
        .INPUT_DIM(HIDDEN_DIM),
        .OUTPUT_DIM(EMBEDDING_DIM),
        .MAX_TOKENS(MAX_TOKENS),
        .BUS_WIDTH(BUS_WIDTH)
    ) ff2_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(ff2_start),
        .done(ff2_done),
        .sequence_length(sequence_length),
        .input_embeddings(ff1_output),
        .output_embeddings(output_embeddings),  // Directly write to output
        .weight_base_addr(ff2_weights_addr),
        .mem_rd_en(ff2_mem_rd_en),
        .mem_rd_addr(ff2_mem_addr),
        .mem_rd_data(mem_rd_data),
        .mem_rd_valid(mem_rd_valid)
    );
    
    // Memory arbiter
    always_comb begin
        // Default: no access
        mem_rd_en = 1'b0;
        mem_rd_addr = 32'h0;
        
        // Priority-based arbitration
        if (ln1_mem_rd_en) begin
            mem_rd_en = ln1_mem_rd_en;
            mem_rd_addr = ln1_mem_addr;
        end else if (qkv_mem_rd_en) begin
            mem_rd_en = qkv_mem_rd_en;
            mem_rd_addr = qkv_mem_addr;
        end else if (attn_mem_rd_en) begin
            mem_rd_en = attn_mem_rd_en;
            mem_rd_addr = attn_mem_addr;
        end else if (out_mem_rd_en) begin
            mem_rd_en = out_mem_rd_en;
            mem_rd_addr = out_mem_addr;
        end else if (ln2_mem_rd_en) begin
            mem_rd_en = ln2_mem_rd_en;
            mem_rd_addr = ln2_mem_addr;
        end else if (ff1_mem_rd_en) begin
            mem_rd_en = ff1_mem_rd_en;
            mem_rd_addr = ff1_mem_addr;
        end else if (ff2_mem_rd_en) begin
            mem_rd_en = ff2_mem_rd_en;
            mem_rd_addr = ff2_mem_addr;
        end
    end
    
    // State machine logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= TL_IDLE;
            weights_loaded <= 1'b0;
            qkv_weights_addr <= '0;
            out_weights_addr <= '0;
            ff1_weights_addr <= '0;
            ff2_weights_addr <= '0;
            
            // Control signals
            layernorm1_start <= 1'b0;
            qkv_project_start <= 1'b0;
            attention_start <= 1'b0;
            layernorm2_start <= 1'b0;
            ff1_start <= 1'b0;
            ff2_start <= 1'b0;
            done <= 1'b0;
            
            // Initialize residual connections
            for (int i = 0; i < MAX_TOKENS; i++) begin
                for (int j = 0; j < EMBEDDING_DIM; j++) begin
                    attention_output[i][j] <= '0;
                    output_embeddings[i][j] <= '0;
                end
            end
        end else begin
            current_state <= next_state;
            
            // Default values for pulse signals
            layernorm1_start <= 1'b0;
            qkv_project_start <= 1'b0;
            attention_start <= 1'b0;
            layernorm2_start <= 1'b0;
            ff1_start <= 1'b0;
            ff2_start <= 1'b0;
            done <= 1'b0;
            
            case (current_state)
                TL_IDLE: begin
                    if (start) begin
                        weights_loaded <= 1'b0;
                        
                        // Calculate weight addresses based on layer index
                        qkv_weights_addr <= weight_base_addr + 32'h2000;
                        out_weights_addr <= weight_base_addr + 32'h4000;
                        ff1_weights_addr <= weight_base_addr + 32'h5000;
                        ff2_weights_addr <= weight_base_addr + 32'h7000;
                    end
                end
                
                TL_LOAD_WEIGHTS: begin
                    // In a real implementation, we might need to load weights
                    // For simulation, we'll assume they're accessible directly
                    weights_loaded <= 1'b1;
                end
                
                TL_LAYER_NORM1: begin
                    if (!layernorm1_done && !layernorm1_start) begin
                        layernorm1_start <= 1'b1;
                    end
                end
                
                TL_QKV_PROJECT: begin
                    if (!qkv_project_done && !qkv_project_start) begin
                        qkv_project_start <= 1'b1;
                    end
                end
                
                TL_ATTENTION: begin
                    if (!attention_done && !attention_start) begin
                        attention_start <= 1'b1;
                    end
                end
                
                TL_ADD_RESIDUAL1: begin
                    // Add residual connection after attention
                    for (int i = 0; i < MAX_TOKENS; i++) begin
                        for (int j = 0; j < EMBEDDING_DIM; j++) begin
                            attention_output[i][j] <= attention_output[i][j] + input_embeddings[i][j];
                        end
                    end
                end
                
                TL_LAYER_NORM2: begin
                    if (!layernorm2_done && !layernorm2_start) begin
                        layernorm2_start <= 1'b1;
                    end
                end
                
                TL_FF1: begin
                    if (!ff1_done && !ff1_start) begin
                        ff1_start <= 1'b1;
                    end
                end
                
                TL_FF2: begin
                    if (!ff2_done && !ff2_start) begin
                        ff2_start <= 1'b1;
                    end
                end
                
                TL_ADD_RESIDUAL2: begin
                    // Add residual connection after feed-forward
                    for (int i = 0; i < MAX_TOKENS; i++) begin
                        for (int j = 0; j < EMBEDDING_DIM; j++) begin
                            output_embeddings[i][j] <= output_embeddings[i][j] + attention_output[i][j];
                        end
                    end
                end
                
                TL_FINISHED: begin
                    done <= 1'b1;
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            TL_IDLE: begin
                if (start) begin
                    next_state = TL_LOAD_WEIGHTS;
                end
            end
            
            TL_LOAD_WEIGHTS: begin
                if (weights_loaded) begin
                    next_state = TL_LAYER_NORM1;
                end
            end
            
            TL_LAYER_NORM1: begin
                if (layernorm1_done) begin
                    next_state = TL_QKV_PROJECT;
                end
            end
            
            TL_QKV_PROJECT: begin
                if (qkv_project_done) begin
                    next_state = TL_ATTENTION;
                end
            end
            
            TL_ATTENTION: begin
                if (attention_done) begin
                    next_state = TL_ADD_RESIDUAL1;
                end
            end
            
            TL_ADD_RESIDUAL1: begin
                next_state = TL_LAYER_NORM2;
            end
            
            TL_LAYER_NORM2: begin
                if (layernorm2_done) begin
                    next_state = TL_FF1;
                end
            end
            
            TL_FF1: begin
                if (ff1_done) begin
                    next_state = TL_FF2;
                end
            end
            
            TL_FF2: begin
                if (ff2_done) begin
                    next_state = TL_ADD_RESIDUAL2;
                end
            end
            
            TL_ADD_RESIDUAL2: begin
                next_state = TL_FINISHED;
            end
            
            TL_FINISHED: begin
                next_state = TL_IDLE;
            end
            
            default: next_state = TL_IDLE;
        endcase
    end

endmodule
