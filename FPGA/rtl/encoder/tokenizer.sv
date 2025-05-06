//-----------------------------------------------------------------------------
// File: tokenizer.sv
// 
// Description: Tokenizer module for RAG-CSD
//              Converts input text to token IDs using WordPiece tokenization
//
// Parameters:
//   MAX_TOKENS  - Maximum number of tokens to process
//   VOCAB_SIZE  - Size of the vocabulary
//   BUS_WIDTH   - AXI bus width
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module tokenizer #(
    parameter int MAX_TOKENS = 128,
    parameter int VOCAB_SIZE = 30522,       // BERT vocabulary size
    parameter int BUS_WIDTH = 512,
    parameter int CHAR_BUF_SIZE = 1024
) (
    // Clock and reset
    input  logic                            clk,
    input  logic                            rst_n,
    
    // Control signals
    input  logic                            start,
    output logic                            done,
    
    // Input text data
    input  logic [BUS_WIDTH-1:0]            text_data,
    input  logic                            text_valid,
    input  logic                            text_last,
    
    // Output tokens
    output logic [MAX_TOKENS-1:0][31:0]     token_ids,
    output logic [$clog2(MAX_TOKENS):0]     sequence_length,
    
    // Memory interface for vocabulary access
    output logic                            mem_rd_en,
    output logic [31:0]                     mem_rd_addr,
    input  logic [BUS_WIDTH-1:0]            mem_rd_data,
    input  logic                            mem_rd_valid
);

    // Tokenizer state machine
    typedef enum logic [3:0] {
        T_IDLE,
        T_LOAD_VOCAB,
        T_BUFFER_TEXT,
        T_TOKENIZE_BASIC,
        T_WORDPIECE,
        T_SPECIAL_TOKENS,
        T_POST_PROCESS,
        T_FINISHED
    } tokenizer_state_t;
    
    tokenizer_state_t current_state, next_state;
    
    // Internal registers and buffers
    logic [7:0] char_buffer [CHAR_BUF_SIZE-1:0];  // Buffer for incoming characters
    logic [$clog2(CHAR_BUF_SIZE):0] char_count;   // Number of characters in buffer
    logic [$clog2(MAX_TOKENS):0] token_idx;       // Current token index
    logic [$clog2(CHAR_BUF_SIZE):0] char_pos;     // Current character position
    logic vocab_loaded;                           // Flag for vocabulary loading
    logic text_buffering_done;                    // Flag for text buffering completion
    
    // Memory interface state
    logic [31:0] vocab_base_addr;                 // Base address of vocabulary in memory
    logic [31:0] mem_words_to_read;               // Words to read from memory
    logic [31:0] mem_words_read;                  // Words read from memory
    
    // Constants
    localparam int CLS_TOKEN_ID = 101;            // [CLS] token ID
    localparam int SEP_TOKEN_ID = 102;            // [SEP] token ID
    localparam int UNK_TOKEN_ID = 100;            // [UNK] token ID
    
    // Initialize state machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= T_IDLE;
            char_count <= '0;
            token_idx <= '0;
            char_pos <= '0;
            sequence_length <= '0;
            mem_rd_en <= 1'b0;
            mem_rd_addr <= '0;
            vocab_loaded <= 1'b0;
            text_buffering_done <= 1'b0;
            mem_words_to_read <= '0;
            mem_words_read <= '0;
            vocab_base_addr <= 32'h2000;  // Default vocabulary base address
            done <= 1'b0;
            
            // Initialize token output
            for (int i = 0; i < MAX_TOKENS; i++) begin
                token_ids[i] <= '0;
            end
        end else begin
            current_state <= next_state;
            
            case (current_state)
                T_IDLE: begin
                    if (start) begin
                        char_count <= '0;
                        token_idx <= '0;
                        char_pos <= '0;
                        sequence_length <= '0;
                        text_buffering_done <= 1'b0;
                        done <= 1'b0;
                    end
                end
                
                T_LOAD_VOCAB: begin
                    // In a real implementation, we would load vocabulary
                    // For simulation, we'll assume vocabulary is pre-loaded
                    vocab_loaded <= 1'b1;
                end
                
                T_BUFFER_TEXT: begin
                    if (text_valid) begin
                        // Buffer the incoming text data
                        for (int i = 0; i < (BUS_WIDTH/8); i++) begin
                            if (char_count < CHAR_BUF_SIZE) begin
                                char_buffer[char_count] <= text_data[i*8 +: 8];
                                char_count <= char_count + 1;
                            end
                        end
                        
                        if (text_last) begin
                            text_buffering_done <= 1'b1;
                        end
                    end
                end
                
                T_TOKENIZE_BASIC: begin
                    // In a real implementation, this would be a complex tokenization logic
                    // For simulation, we'll use a simplified approach that just creates tokens
                    // based on spaces and punctuation
                    
                    // Process one character at a time
                    if (char_pos < char_count) begin
                        logic [7:0] current_char = char_buffer[char_pos];
                        
                        // Check if it's a space or punctuation
                        if (current_char == 8'h20 || current_char == 8'h2E || 
                            current_char == 8'h2C || current_char == 8'h21 || 
                            current_char == 8'h3F) begin
                            
                            // End of token, create a new token ID
                            if (token_idx < MAX_TOKENS && char_pos > 0) begin
                                // In real implementation, would lookup vocab here
                                // For simulation, we'll use a simple mapping
                                token_ids[token_idx] <= 1000 + token_idx;
                                token_idx <= token_idx + 1;
                            end
                        end
                        
                        char_pos <= char_pos + 1;
                    end
                end
                
                T_WORDPIECE: begin
                    // In a real implementation, this would handle WordPiece tokenization
                    // For simulation, we'll skip this step
                end
                
                T_SPECIAL_TOKENS: begin
                    // Add special tokens (CLS at beginning, SEP at end)
                    if (token_idx < MAX_TOKENS - 1) begin
                        // Shift tokens to make room for CLS
                        for (int i = token_idx; i > 0; i--) begin
                            token_ids[i] <= token_ids[i-1];
                        end
                        
                        // Add CLS and SEP tokens
                        token_ids[0] <= CLS_TOKEN_ID;
                        token_ids[token_idx+1] <= SEP_TOKEN_ID;
                        
                        // Update token count
                        token_idx <= token_idx + 2;
                    end
                end
                
                T_POST_PROCESS: begin
                    // Set final sequence length
                    sequence_length <= token_idx;
                end
                
                T_FINISHED: begin
                    done <= 1'b1;
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            T_IDLE: begin
                if (start) begin
                    if (vocab_loaded) begin
                        next_state = T_BUFFER_TEXT;
                    end else begin
                        next_state = T_LOAD_VOCAB;
                    end
                end
            end
            
            T_LOAD_VOCAB: begin
                if (vocab_loaded) begin
                    next_state = T_BUFFER_TEXT;
                end
            end
            
            T_BUFFER_TEXT: begin
                if (text_buffering_done) begin
                    next_state = T_TOKENIZE_BASIC;
                end
            end
            
            T_TOKENIZE_BASIC: begin
                if (char_pos >= char_count) begin
                    next_state = T_WORDPIECE;
                end
            end
            
            T_WORDPIECE: begin
                next_state = T_SPECIAL_TOKENS;
            end
            
            T_SPECIAL_TOKENS: begin
                next_state = T_POST_PROCESS;
            end
            
            T_POST_PROCESS: begin
                next_state = T_FINISHED;
            end
            
            T_FINISHED: begin
                next_state = T_IDLE;
            end
            
            default: next_state = T_IDLE;
        endcase
    end

endmodule
