//-----------------------------------------------------------------------------
// File: embedding_lookup.sv
// 
// Description: Embedding Lookup Module for RAG-CSD
//              Converts token IDs to embedding vectors
//
// Parameters:
//   EMBEDDING_DIM  - Embedding dimension
//   MAX_TOKENS     - Maximum tokens per query
//   VOCAB_SIZE     - Vocabulary size
//   BUS_WIDTH      - AXI bus width
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module embedding_lookup #(
    parameter int EMBEDDING_DIM = 384,
    parameter int MAX_TOKENS = 128,
    parameter int VOCAB_SIZE = 30522,
    parameter int BUS_WIDTH = 512
) (
    // Clock and reset
    input  logic                                  clk,
    input  logic                                  rst_n,
    
    // Control signals
    input  logic                                  start,
    output logic                                  done,
    
    // Input tokens
    input  logic [MAX_TOKENS-1:0][31:0]           token_ids,
    input  logic [$clog2(MAX_TOKENS):0]           sequence_length,
    
    // Output embeddings
    output logic [MAX_TOKENS-1:0][EMBEDDING_DIM-1:0][31:0] token_embeddings,
    
    // Weight base address
    input  logic [31:0]                           weight_base_addr,
    
    // Memory interface
    output logic                                  mem_rd_en,
    output logic [31:0]                           mem_rd_addr,
    input  logic [BUS_WIDTH-1:0]                  mem_rd_data,
    input  logic                                  mem_rd_valid
);

    // Internal state machine
    typedef enum logic [2:0] {
        EL_IDLE,
        EL_LOAD_EMBEDDING_TABLE,
        EL_PROCESS_TOKEN,
        EL_FETCH_EMBEDDING,
        EL_FINISHED
    } embed_lookup_state_t;
    
    embed_lookup_state_t current_state, next_state;
    
    // Internal counters and registers
    logic [$clog2(MAX_TOKENS):0] token_ptr;         // Current token being processed
    logic [$clog2(EMBEDDING_DIM):0] embed_offset;   // Current embedding offset
    logic [31:0] embedding_table_addr;              // Address of embedding table
    
    // Memory interface state
    logic [31:0] mem_words_to_read;                 // Words to read from memory
    logic [31:0] mem_words_read;                    // Words read from memory
    logic mem_read_complete;                        // Memory read completion flag
    
    // State machine logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= EL_IDLE;
            token_ptr <= '0;
            embed_offset <= '0;
            mem_rd_en <= 1'b0;
            mem_rd_addr <= '0;
            mem_words_to_read <= '0;
            mem_words_read <= '0;
            mem_read_complete <= 1'b0;
            embedding_table_addr <= '0;
            done <= 1'b0;
            
            // Initialize token embeddings
            for (int i = 0; i < MAX_TOKENS; i++) begin
                for (int j = 0; j < EMBEDDING_DIM; j++) begin
                    token_embeddings[i][j] <= '0;
                end
            end
        end else begin
            current_state <= next_state;
            
            case (current_state)
                EL_IDLE: begin
                    if (start) begin
                        token_ptr <= '0;
                        embed_offset <= '0;
                        mem_read_complete <= 1'b0;
                        done <= 1'b0;
                        
                        // Calculate embedding table address
                        embedding_table_addr <= weight_base_addr;
                    end
                end
                
                EL_LOAD_EMBEDDING_TABLE: begin
                    // In a real implementation, we might need to load the embedding table
                    // For simulation, we'll assume it's accessible directly
                    mem_read_complete <= 1'b1;
                end
                
                EL_PROCESS_TOKEN: begin
                    // Start processing next token if not at end of sequence
                    if (token_ptr < sequence_length) begin
                        embed_offset <= '0;
                        mem_read_complete <= 1'b0;
                    end
                end
                
                EL_FETCH_EMBEDDING: begin
                    // Fetch embedding for current token
                    if (!mem_read_complete) begin
                        // Calculate memory address for this token's embedding
                        mem_rd_en <= 1'b1;
                        mem_rd_addr <= embedding_table_addr + 
                                      (token_ids[token_ptr] * EMBEDDING_DIM * 4) + 
                                      (embed_offset * 4);
                        
                        // Determine how many words to read based on bus width
                        mem_words_to_read <= (EMBEDDING_DIM * 4 + BUS_WIDTH - 1) / BUS_WIDTH;
                        
                        // Increment embedding offset for next read
                        if (embed_offset < EMBEDDING_DIM) begin
                            embed_offset <= embed_offset + (BUS_WIDTH / 32);
                        end
                        
                        if (embed_offset >= EMBEDDING_DIM - (BUS_WIDTH / 32)) begin
                            mem_read_complete <= 1'b1;
                        end
                    end
                    
                    // Process memory read data if valid
                    if (mem_rd_valid) begin
                        // Extract embedding values from memory data
                        for (int i = 0; i < (BUS_WIDTH / 32); i++) begin
                            if ((embed_offset - (BUS_WIDTH / 32)) + i < EMBEDDING_DIM) begin
                                token_embeddings[token_ptr][(embed_offset - (BUS_WIDTH / 32)) + i] <= 
                                    mem_rd_data[i*32 +: 32];
                            end
                        end
                    end
                    
                    // If memory read is complete, move to next token
                    if (mem_read_complete) begin
                        mem_rd_en <= 1'b0;
                        token_ptr <= token_ptr + 1;
                    end
                end
                
                EL_FINISHED: begin
                    done <= 1'b1;
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            EL_IDLE: begin
                if (start) begin
                    next_state = EL_LOAD_EMBEDDING_TABLE;
                end
            end
            
            EL_LOAD_EMBEDDING_TABLE: begin
                if (mem_read_complete) begin
                    next_state = EL_PROCESS_TOKEN;
                end
            end
            
            EL_PROCESS_TOKEN: begin
                if (token_ptr < sequence_length) begin
                    next_state = EL_FETCH_EMBEDDING;
                end else begin
                    next_state = EL_FINISHED;
                end
            end
            
            EL_FETCH_EMBEDDING: begin
                if (mem_read_complete) begin
                    next_state = EL_PROCESS_TOKEN;
                end
            end
            
            EL_FINISHED: begin
                next_state = EL_IDLE;
            end
            
            default: next_state = EL_IDLE;
        endcase
    end

endmodule
