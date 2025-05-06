//-----------------------------------------------------------------------------
// File: encoder.sv
// 
// Description: Encoder module for RAG-CSD
//              Implements a lightweight transformer encoder for generating 
//              embeddings based on MiniLM architecture
//
// Parameters:
//   EMBEDDING_DIM  - Embedding dimension
//   MAX_TOKENS     - Maximum tokens per query
//   NUM_HEADS      - Number of attention heads
//   NUM_LAYERS     - Number of transformer layers
//   HIDDEN_DIM     - Hidden dimension in feed-forward
//   VOCAB_SIZE     - Vocabulary size
//   BUS_WIDTH      - AXI bus width
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module encoder #(
    parameter int EMBEDDING_DIM = 384,      // Default for MiniLM-L6
    parameter int MAX_TOKENS = 128,         // Maximum tokens per query
    parameter int NUM_HEADS = 6,            // Number of attention heads (384/64)
    parameter int NUM_LAYERS = 6,           // Number of transformer layers
    parameter int HIDDEN_DIM = 1536,        // Hidden dimension in feed-forward
    parameter int VOCAB_SIZE = 30522,       // Vocabulary size (BERT vocabulary)
    parameter int BUS_WIDTH = 512,          // AXI bus width
    parameter int HEAD_DIM = EMBEDDING_DIM / NUM_HEADS // Dimension per head
) (
    // Clock and reset
    input  logic                            clk,
    input  logic                            rst_n,
    
    // Control signals
    input  logic                            start,
    output logic                            done,
    
    // Input query data
    input  logic [BUS_WIDTH-1:0]            query_data,
    input  logic                            query_valid,
    input  logic                            query_last,
    
    // Output embedding
    output logic [EMBEDDING_DIM-1:0][31:0]  embedding,
    
    // Memory interface
    output logic                            mem_rd_en,
    output logic [31:0]                     mem_rd_addr,
    input  logic [BUS_WIDTH-1:0]            mem_rd_data,
    input  logic                            mem_rd_valid
);

    // Internal state machine
    typedef enum logic [3:0] {
        IDLE,
        LOAD_WEIGHTS,
        TOKENIZE,
        EMBEDDING_LOOKUP,
        LAYER_PROCESSING,
        POOLING,
        NORMALIZE,
        FINISHED
    } encoder_state_t;
    
    encoder_state_t current_state, next_state;
    
    // Internal counters and registers
    logic [$clog2(MAX_TOKENS)-1:0] token_count;
    logic [$clog2(NUM_LAYERS)-1:0] layer_idx;
    logic weights_loaded;
    
    // Tokenizer signals
    logic tokenize_start, tokenize_done;
    logic [MAX_TOKENS-1:0][31:0] token_ids;
    logic [$clog2(MAX_TOKENS):0] sequence_length;
    
    // Embedding lookup signals
    logic embedding_lookup_start, embedding_lookup_done;
    logic [MAX_TOKENS-1:0][EMBEDDING_DIM-1:0][31:0] token_embeddings;
    
    // Layer processing signals
    logic layer_start, layer_done;
    logic [MAX_TOKENS-1:0][EMBEDDING_DIM-1:0][31:0] layer_input, layer_output;
    
    // Pooling and normalization signals
    logic pooling_start, pooling_done;
    logic normalize_start, normalize_done;
    logic [EMBEDDING_DIM-1:0][31:0] pooled_output;
    
    // Memory arbitration signals
    logic tokenizer_mem_rd_en, embedding_lookup_mem_rd_en, layer_mem_rd_en;
    logic [31:0] tokenizer_mem_addr, embedding_lookup_mem_addr, layer_mem_addr;
    logic [31:0] weight_base_addr;
    
    // Instantiate tokenizer module
    tokenizer #(
        .MAX_TOKENS(MAX_TOKENS),
        .VOCAB_SIZE(VOCAB_SIZE),
        .BUS_WIDTH(BUS_WIDTH)
    ) tokenizer_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(tokenize_start),
        .done(tokenize_done),
        .text_data(query_data),
        .text_valid(query_valid),
        .text_last(query_last),
        .token_ids(token_ids),
        .sequence_length(sequence_length),
        // Memory interface
        .mem_rd_en(tokenizer_mem_rd_en),
        .mem_rd_addr(tokenizer_mem_addr),
        .mem_rd_data(mem_rd_data),
        .mem_rd_valid(mem_rd_valid)
    );
    
    // Embedding lookup
    embedding_lookup #(
        .EMBEDDING_DIM(EMBEDDING_DIM),
        .MAX_TOKENS(MAX_TOKENS),
        .VOCAB_SIZE(VOCAB_SIZE),
        .BUS_WIDTH(BUS_WIDTH)
    ) embedding_lookup_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(embedding_lookup_start),
        .done(embedding_lookup_done),
        .token_ids(token_ids),
        .sequence_length(sequence_length),
        .token_embeddings(token_embeddings),
        .weight_base_addr(weight_base_addr),
        // Memory interface
        .mem_rd_en(embedding_lookup_mem_rd_en),
        .mem_rd_addr(embedding_lookup_mem_addr),
        .mem_rd_data(mem_rd_data),
        .mem_rd_valid(mem_rd_valid)
    );
    
    // Instantiate transformer layer
    transformer_layer #(
        .EMBEDDING_DIM(EMBEDDING_DIM),
        .MAX_TOKENS(MAX_TOKENS),
        .HEAD_DIM(HEAD_DIM),
        .NUM_HEADS(NUM_HEADS),
        .HIDDEN_DIM(HIDDEN_DIM),
        .BUS_WIDTH(BUS_WIDTH)
    ) transformer_layer_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(layer_start),
        .done(layer_done),
        .layer_idx(layer_idx),
        .sequence_length(sequence_length),
        .input_embeddings(layer_input),
        .output_embeddings(layer_output),
        .weight_base_addr(weight_base_addr + 32'h10000 + (layer_idx * 32'h8000)),
        // Memory interface
        .mem_rd_en(layer_mem_rd_en),
        .mem_rd_addr(layer_mem_addr),
        .mem_rd_data(mem_rd_data),
        .mem_rd_valid(mem_rd_valid)
    );
    
    // Mean pooling module
    mean_pooler #(
        .EMBEDDING_DIM(EMBEDDING_DIM),
        .MAX_TOKENS(MAX_TOKENS)
    ) mean_pooler_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(pooling_start),
        .done(pooling_done),
        .sequence_length(sequence_length),
        .token_embeddings(layer_output),
        .pooled_embedding(pooled_output)
    );
    
    // Embedding normalizer (L2 norm)
    embedding_normalizer #(
        .EMBEDDING_DIM(EMBEDDING_DIM)
    ) normalizer_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(normalize_start),
        .done(normalize_done),
        .input_embedding(pooled_output),
        .normalized_embedding(embedding)
    );
    
    // Memory arbiter for encoder components
    always_comb begin
        // Default: no access
        mem_rd_en = 1'b0;
        mem_rd_addr = 32'h0;
        
        // Priority-based arbitration
        if (tokenizer_mem_rd_en) begin
            mem_rd_en = tokenizer_mem_rd_en;
            mem_rd_addr = tokenizer_mem_addr;
        end else if (embedding_lookup_mem_rd_en) begin
            mem_rd_en = embedding_lookup_mem_rd_en;
            mem_rd_addr = embedding_lookup_mem_addr;
        end else if (layer_mem_rd_en) begin
            mem_rd_en = layer_mem_rd_en;
            mem_rd_addr = layer_mem_addr;
        end
    end
    
    // State machine for controlling the encoder pipeline
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
            layer_idx <= '0;
            weight_base_addr <= 32'h1000;  // Default weight base address
            weights_loaded <= 1'b0;
            
            // Clear control signals
            tokenize_start <= 1'b0;
            embedding_lookup_start <= 1'b0;
            layer_start <= 1'b0;
            pooling_start <= 1'b0;
            normalize_start <= 1'b0;
            done <= 1'b0;
        end else begin
            current_state <= next_state;
            
            // Default values for pulse signals
            tokenize_start <= 1'b0;
            embedding_lookup_start <= 1'b0;
            layer_start <= 1'b0;
            pooling_start <= 1'b0;
            normalize_start <= 1'b0;
            done <= 1'b0;
            
            case (current_state)
                IDLE: begin
                    if (start) begin
                        layer_idx <= '0;
                        weights_loaded <= 1'b0;
                    end
                end
                
                LOAD_WEIGHTS: begin
                    // In real implementation, we would load weights here
                    // For simulation, we'll assume weights are pre-loaded
                    weights_loaded <= 1'b1;
                end
                
                TOKENIZE: begin
                    if (!tokenize_done && !tokenize_start) begin
                        tokenize_start <= 1'b1;
                    end
                end
                
                EMBEDDING_LOOKUP: begin
                    if (!embedding_lookup_done && !embedding_lookup_start) begin
                        embedding_lookup_start <= 1'b1;
                    end
                end
                
                LAYER_PROCESSING: begin
                    if (!layer_done && !layer_start) begin
                        layer_start <= 1'b1;
                    end
                    
                    if (layer_done) begin
                        if (layer_idx < NUM_LAYERS - 1) begin
                            layer_idx <= layer_idx + 1;
                            layer_start <= 1'b1;
                        end
                    end
                end
                
                POOLING: begin
                    if (!pooling_done && !pooling_start) begin
                        pooling_start <= 1'b1;
                    end
                end
                
                NORMALIZE: begin
                    if (!normalize_done && !normalize_start) begin
                        normalize_start <= 1'b1;
                    end
                end
                
                FINISHED: begin
                    done <= 1'b1;
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            IDLE: begin
                if (start) begin
                    next_state = LOAD_WEIGHTS;
                end
            end
            
            LOAD_WEIGHTS: begin
                if (weights_loaded) begin
                    next_state = TOKENIZE;
                end
            end
            
            TOKENIZE: begin
                if (tokenize_done) begin
                    next_state = EMBEDDING_LOOKUP;
                end
            end
            
            EMBEDDING_LOOKUP: begin
                if (embedding_lookup_done) begin
                    next_state = LAYER_PROCESSING;
                end
            end
            
            LAYER_PROCESSING: begin
                if (layer_done && layer_idx == NUM_LAYERS - 1) begin
                    next_state = POOLING;
                end
            end
            
            POOLING: begin
                if (pooling_done) begin
                    next_state = NORMALIZE;
                end
            end
            
            NORMALIZE: begin
                if (normalize_done) begin
                    next_state = FINISHED;
                end
            end
            
            FINISHED: begin
                next_state = IDLE;
            end
            
            default: next_state = IDLE;
        endcase
    end
    
    // Update layer input based on current layer
    always_comb begin
        if (layer_idx == 0) begin
            layer_input = token_embeddings;
        end else begin
            layer_input = layer_output;
        end
    end

endmodule
