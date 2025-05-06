//-----------------------------------------------------------------------------
// File: augmentation.sv
// 
// Description: Augmentation Module for RAG-CSD
//              Combines query with retrieved documents to create augmented query
//
// Parameters:
//   MAX_SEQUENCE_LEN  - Maximum sequence length
//   TOP_K             - Number of retrieved documents
//   EMBEDDING_DIM     - Embedding dimension
//   BUS_WIDTH         - AXI bus width
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module augmentation #(
    parameter int MAX_SEQUENCE_LEN = 512,
    parameter int TOP_K = 5,
    parameter int EMBEDDING_DIM = 384,
    parameter int BUS_WIDTH = 512
) (
    // Clock and reset
    input  logic                                clk,
    input  logic                                rst_n,
    
    // Control signals
    input  logic                                start,
    output logic                                done,
    
    // Query data
    input  logic [BUS_WIDTH-1:0]                query_data,
    
    // Retrieved documents
    input  logic [TOP_K-1:0][EMBEDDING_DIM-1:0][31:0] retrieved_vectors,
    input  logic [TOP_K-1:0][31:0]              similarity_scores,
    input  logic [TOP_K-1:0][31:0]              doc_indices,
    
    // Memory interface for document content
    output logic                                mem_rd_en,
    output logic [31:0]                         mem_rd_addr,
    input  logic [BUS_WIDTH-1:0]                mem_rd_data,
    input  logic                                mem_rd_valid,
    
    // Augmented output
    output logic [BUS_WIDTH-1:0]                augmented_data,
    output logic                                augmented_valid,
    output logic                                augmented_last
);

    // Augmentation strategy
    typedef enum logic [1:0] {
        CONCAT_STRATEGY = 2'b00,
        TEMPLATE_STRATEGY = 2'b01,
        WEIGHTED_STRATEGY = 2'b10
    } augmentation_strategy_t;
    
    // Augmentation states
    typedef enum logic [3:0] {
        A_IDLE,
        A_LOAD_CONFIG,
        A_BUFFER_QUERY,
        A_FETCH_DOCS,
        A_PROCESS_DOCS,
        A_DEDUPLICATE,
        A_BUILD_OUTPUT,
        A_STREAM_OUTPUT,
        A_FINISHED
    } augmentation_state_t;
    
    augmentation_state_t current_state, next_state;
    
    // Internal registers and buffers
    logic [1:0] augmentation_strategy;              // Strategy selection
    logic [$clog2(TOP_K)-1:0] doc_idx;              // Document index being processed
    logic [31:0] doc_content_addr;                  // Base address for document content
    
    // Character buffers
    logic [7:0] query_buffer [MAX_SEQUENCE_LEN-1:0];
    logic [31:0] query_length;
    
    // Document content buffers
    logic [7:0] doc_buffers [TOP_K-1:0][MAX_SEQUENCE_LEN-1:0];
    logic [31:0] doc_lengths [TOP_K-1:0];
    
    // Deduplication tracking
    logic [TOP_K-1:0] doc_included;
    
    // Output buffer and streaming control
    logic [7:0] output_buffer [MAX_SEQUENCE_LEN*(TOP_K+1)-1:0];
    logic [31:0] output_size;
    logic [31:0] output_position;
    
    // Memory interface state
    logic [31:0] mem_words_to_read;
    logic [31:0] mem_words_read;
    logic config_loaded;
    logic doc_loaded;
    
    // Text buffer manager for handling efficient text processing
    logic buffer_start, buffer_done;
    logic [31:0] buffer_length;
    
    text_buffer_manager #(
        .MAX_SEQUENCE_LEN(MAX_SEQUENCE_LEN),
        .TOP_K(TOP_K)
    ) buffer_mgr (
        .clk(clk),
        .rst_n(rst_n),
        .start(buffer_start),
        .done(buffer_done),
        .strategy(augmentation_strategy),
        .query_buffer(query_buffer),
        .query_length(query_length),
        .doc_buffers(doc_buffers),
        .doc_lengths(doc_lengths),
        .doc_included(doc_included),
        .similarity_scores(similarity_scores),
        .output_buffer(output_buffer),
        .output_length(buffer_length)
    );
    
    // Deduplication module to remove duplicate content
    logic dedup_start, dedup_done;
    
    deduplication #(
        .MAX_SEQUENCE_LEN(MAX_SEQUENCE_LEN),
        .TOP_K(TOP_K)
    ) dedup (
        .clk(clk),
        .rst_n(rst_n),
        .start(dedup_start),
        .done(dedup_done),
        .doc_buffers(doc_buffers),
        .doc_lengths(doc_lengths),
        .doc_included(doc_included)
    );
    
    // Template formatter for template-based augmentation
    logic template_start, template_done;
    logic [31:0] template_length;
    
    template_formatter #(
        .MAX_SEQUENCE_LEN(MAX_SEQUENCE_LEN),
        .TOP_K(TOP_K)
    ) template_fmt (
        .clk(clk),
        .rst_n(rst_n),
        .start(template_start),
        .done(template_done),
        .query_buffer(query_buffer),
        .query_length(query_length),
        .doc_buffers(doc_buffers),
        .doc_lengths(doc_lengths),
        .doc_included(doc_included),
        .similarity_scores(similarity_scores),
        .output_buffer(output_buffer),
        .output_length(template_length)
    );
    
    // State machine logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= A_IDLE;
            doc_idx <= '0;
            doc_content_addr <= '0;
            query_length <= '0;
            output_size <= '0;
            output_position <= '0;
            augmentation_strategy <= CONCAT_STRATEGY;
            config_loaded <= 1'b0;
            doc_loaded <= 1'b0;
            mem_rd_en <= 1'b0;
            mem_rd_addr <= '0;
            mem_words_to_read <= '0;
            mem_words_read <= '0;
            buffer_start <= 1'b0;
            dedup_start <= 1'b0;
            template_start <= 1'b0;
            augmented_valid <= 1'b0;
            augmented_last <= 1'b0;
            done <= 1'b0;
            
            // Initialize doc_included flags
            for (int i = 0; i < TOP_K; i++) begin
                doc_included[i] <= 1'b0;
                doc_lengths[i] <= '0;
            end
        end else begin
            current_state <= next_state;
            
            // Default for pulse signals
            buffer_start <= 1'b0;
            dedup_start <= 1'b0;
            template_start <= 1'b0;
            augmented_valid <= 1'b0;
            augmented_last <= 1'b0;
            
            case (current_state)
                A_IDLE: begin
                    if (start) begin
                        doc_idx <= '0;
                        query_length <= '0;
                        output_size <= '0;
                        output_position <= '0;
                        config_loaded <= 1'b0;
                        doc_loaded <= 1'b0;
                        done <= 1'b0;
                        
                        // Reset doc_included flags
                        for (int i = 0; i < TOP_K; i++) begin
                            doc_included[i] <= 1'b0;
                            doc_lengths[i] <= '0;
                        end
                    end
                end
                
                A_LOAD_CONFIG: begin
                    if (!config_loaded) begin
                        mem_rd_en <= 1'b1;
                        mem_rd_addr <= 32'h2000; // Assume config at this address
                        
                        if (mem_rd_valid) begin
                            augmentation_strategy <= mem_rd_data[1:0];
                            doc_content_addr <= mem_rd_data[63:32];
                            config_loaded <= 1'b1;
                            mem_rd_en <= 1'b0;
                        end
                    end
                end
                
                A_BUFFER_QUERY: begin
                    // Buffer the query data
                    for (int i = 0; i < (BUS_WIDTH/8); i++) begin
                        if (query_length < MAX_SEQUENCE_LEN) begin
                            query_buffer[query_length] <= query_data[i*8 +: 8];
                            query_length <= query_length + 1;
                        end
                    end
                end
                
                A_FETCH_DOCS: begin
                    if (doc_idx < TOP_K) begin
                        if (!doc_loaded) begin
                            // Start fetching document content
                            mem_rd_en <= 1'b1;
                            mem_rd_addr <= doc_content_addr + (doc_indices[doc_idx] * MAX_SEQUENCE_LEN);
                            mem_words_to_read <= (MAX_SEQUENCE_LEN + BUS_WIDTH/8 - 1) / (BUS_WIDTH/8);
                            mem_words_read <= '0;
                            
                            if (mem_rd_valid) begin
                                // Buffer document content
                                for (int i = 0; i < (BUS_WIDTH/8); i++) begin
                                    int pos = mem_words_read * (BUS_WIDTH/8) + i;
                                    if (pos < MAX_SEQUENCE_LEN) begin
                                        doc_buffers[doc_idx][pos] <= mem_rd_data[i*8 +: 8];
                                        
                                        // Update document length if non-null character
                                        if (mem_rd_data[i*8 +: 8] != 8'h00) begin
                                            doc_lengths[doc_idx] <= pos + 1;
                                        end
                                    end
                                end
                                
                                mem_words_read <= mem_words_read + 1;
                                mem_rd_addr <= mem_rd_addr + (BUS_WIDTH/8);
                                
                                if (mem_words_read == mem_words_to_read - 1) begin
                                    mem_rd_en <= 1'b0;
                                    doc_loaded <= 1'b1;
                                end
                            end
                        end
                    end
                end
                
                A_PROCESS_DOCS: begin
                    // Mark current document as included
                    doc_included[doc_idx] <= 1'b1;
                    
                    // Prepare for next document
                    if (doc_idx < TOP_K - 1) begin
                        doc_idx <= doc_idx + 1;
                        doc_loaded <= 1'b0;
                    end
                end
                
                A_DEDUPLICATE: begin
                    if (!dedup_done && !dedup_start) begin
                        dedup_start <= 1'b1;
                    end
                end
                
                A_BUILD_OUTPUT: begin
                    case (augmentation_strategy)
                        CONCAT_STRATEGY: begin
                            if (!buffer_done && !buffer_start) begin
                                buffer_start <= 1'b1;
                            end
                            
                            if (buffer_done) begin
                                output_size <= buffer_length;
                            end
                        end
                        
                        TEMPLATE_STRATEGY: begin
                            if (!template_done && !template_start) begin
                                template_start <= 1'b1;
                            end
                            
                            if (template_done) begin
                                output_size <= template_length;
                            end
                        end
                        
                        WEIGHTED_STRATEGY: begin
                            if (!buffer_done && !buffer_start) begin
                                buffer_start <= 1'b1;
                            end
                            
                            if (buffer_done) begin
                                output_size <= buffer_length;
                            end
                        end
                        
                        default: begin
                            // Default to simple concat
                            if (!buffer_done && !buffer_start) begin
                                buffer_start <= 1'b1;
                            end
                            
                            if (buffer_done) begin
                                output_size <= buffer_length;
                            end
                        end
                    endcase
                end
                
                A_STREAM_OUTPUT: begin
                    // Stream output in BUS_WIDTH chunks
                    augmented_valid <= 1'b1;
                    
                    // Fill output data from buffer
                    for (int i = 0; i < (BUS_WIDTH/8); i++) begin
                        if (output_position + i < output_size) begin
                            augmented_data[i*8 +: 8] <= output_buffer[output_position + i];
                        end else begin
                            augmented_data[i*8 +: 8] <= 8'h00; // Pad with zeros
                        end
                    end
                    
                    // Update position for next chunk
                    output_position <= output_position + (BUS_WIDTH/8);
                    
                    // Check if this is the last chunk
                    if (output_position + (BUS_WIDTH/8) >= output_size) begin
                        augmented_last <= 1'b1;
                    end
                end
                
                A_FINISHED: begin
                    done <= 1'b1;
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            A_IDLE: begin
                if (start) begin
                    next_state = A_LOAD_CONFIG;
                end
            end
            
            A_LOAD_CONFIG: begin
                if (config_loaded) begin
                    next_state = A_BUFFER_QUERY;
                end
            end
            
            A_BUFFER_QUERY: begin
                next_state = A_FETCH_DOCS;
            end
            
            A_FETCH_DOCS: begin
                if (doc_loaded) begin
                    next_state = A_PROCESS_DOCS;
                end
            end
            
            A_PROCESS_DOCS: begin
                if (doc_idx == TOP_K - 1) begin
                    next_state = A_DEDUPLICATE;
                end else begin
                    next_state = A_FETCH_DOCS;
                end
            end
            
            A_DEDUPLICATE: begin
                if (dedup_done) begin
                    next_state = A_BUILD_OUTPUT;
                end
            end
            
            A_BUILD_OUTPUT: begin
                if ((augmentation_strategy == CONCAT_STRATEGY && buffer_done) ||
                    (augmentation_strategy == TEMPLATE_STRATEGY && template_done) ||
                    (augmentation_strategy == WEIGHTED_STRATEGY && buffer_done)) begin
                    next_state = A_STREAM_OUTPUT;
                end
            end
            
            A_STREAM_OUTPUT: begin
                if (augmented_last) begin
                    next_state = A_FINISHED;
                end
            end
            
            A_FINISHED: begin
                next_state = A_IDLE;
            end
            
            default: next_state = A_IDLE;
        endcase
    end

endmodule
