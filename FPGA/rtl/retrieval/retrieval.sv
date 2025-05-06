//-----------------------------------------------------------------------------
// File: retrieval.sv
// 
// Description: Retrieval Module for RAG-CSD
//              Implements efficient vector similarity search
//
// Parameters:
//   EMBEDDING_DIM      - Embedding dimension
//   TOP_K              - Number of results to retrieve
//   VECTOR_CACHE_SIZE  - Size of vector cache
//   BUS_WIDTH          - AXI bus width
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module retrieval #(
    parameter int EMBEDDING_DIM = 384,
    parameter int TOP_K = 5,
    parameter int VECTOR_CACHE_SIZE = 1024,
    parameter int BUS_WIDTH = 512
) (
    // Clock and reset
    input  logic                                clk,
    input  logic                                rst_n,
    
    // Control signals
    input  logic                                start,
    output logic                                done,
    
    // Configuration
    input  logic [1:0]                          similarity_metric, // 0=cosine, 1=dot, 2=euclidean
    input  logic [31:0]                         config_top_k,     // Configurable top-k
    
    // Query embedding
    input  logic [EMBEDDING_DIM-1:0][31:0]      query_embedding,
    
    // Memory interface for vector database access
    output logic                                mem_rd_en,
    output logic [31:0]                         mem_rd_addr,
    input  logic [BUS_WIDTH-1:0]                mem_rd_data,
    input  logic                                mem_rd_valid,
    
    // Retrieved results
    output logic [TOP_K-1:0][EMBEDDING_DIM-1:0][31:0] retrieved_vectors,
    output logic [TOP_K-1:0][31:0]              similarity_scores,
    output logic [TOP_K-1:0][31:0]              doc_indices
);

    // Retrieval states
    typedef enum logic [3:0] {
        R_IDLE,
        R_LOAD_DB_INFO,
        R_FETCH_VECTOR,
        R_COMPUTE_SIMILARITY,
        R_UPDATE_TOPK,
        R_LOAD_RESULTS,
        R_FINISHED
    } retrieval_state_t;
    
    retrieval_state_t current_state, next_state;
    
    // Internal registers
    logic [31:0] db_size;             // Number of vectors in database
    logic [31:0] vector_idx;          // Current vector being processed
    logic [31:0] db_addr_start;       // Start address of vector database
    logic [31:0] vector_stride;       // Bytes per vector
    logic [31:0] metadata_addr_start; // Start address of metadata
    logic [31:0] actual_top_k;        // Actual top-k value to use
    
    // Vector being currently processed
    logic [EMBEDDING_DIM-1:0][31:0] current_vector;
    logic [31:0] current_doc_idx;
    
    // Memory interface state
    logic [31:0] mem_words_to_read;
    logic [31:0] mem_words_read;
    logic db_info_loaded;
    logic mem_read_complete;
    
    // Similarity computation
    logic sim_start, sim_done;
    logic [31:0] similarity_score;
    
    similarity_computer #(
        .EMBEDDING_DIM(EMBEDDING_DIM)
    ) sim_comp (
        .clk(clk),
        .rst_n(rst_n),
        .start(sim_start),
        .done(sim_done),
        .metric(similarity_metric),
        .vec_a(query_embedding),
        .vec_b(current_vector),
        .similarity(similarity_score)
    );
    
    // Top-K queue
    logic pq_push, pq_done;
    logic [31:0] pq_push_similarity;
    logic [31:0] pq_push_index;
    
    top_k_priority_queue #(
        .K(TOP_K),
        .EMBEDDING_DIM(EMBEDDING_DIM)
    ) topk_queue (
        .clk(clk),
        .rst_n(rst_n),
        .push(pq_push),
        .done(pq_done),
        .similarity(pq_push_similarity),
        .doc_index(pq_push_index),
        .vector(current_vector),
        .topk_similarities(similarity_scores),
        .topk_indices(doc_indices),
        .topk_vectors(retrieved_vectors)
    );
    
    // Vector cache for better performance
    logic cache_hit;
    logic [EMBEDDING_DIM-1:0][31:0] cached_vector;
    logic [31:0] cached_doc_idx;
    
    vector_cache #(
        .EMBEDDING_DIM(EMBEDDING_DIM),
        .CACHE_SIZE(VECTOR_CACHE_SIZE)
    ) vec_cache (
        .clk(clk),
        .rst_n(rst_n),
        .lookup_idx(vector_idx),
        .hit(cache_hit),
        .vector(cached_vector),
        .doc_idx(cached_doc_idx),
        .store(mem_read_complete),
        .store_idx(vector_idx),
        .store_vector(current_vector),
        .store_doc_idx(current_doc_idx)
    );
    
    // State machine logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= R_IDLE;
            vector_idx <= '0;
            db_size <= '0;
            db_addr_start <= '0;
            vector_stride <= '0;
            metadata_addr_start <= '0;
            actual_top_k <= TOP_K;
            db_info_loaded <= 1'b0;
            mem_read_complete <= 1'b0;
            mem_rd_en <= 1'b0;
            mem_rd_addr <= '0;
            mem_words_to_read <= '0;
            mem_words_read <= '0;
            sim_start <= 1'b0;
            pq_push <= 1'b0;
            pq_push_similarity <= '0;
            pq_push_index <= '0;
            done <= 1'b0;
        end else begin
            current_state <= next_state;
            
            // Default values for pulse signals
            sim_start <= 1'b0;
            pq_push <= 1'b0;
            
            case (current_state)
                R_IDLE: begin
                    if (start) begin
                        vector_idx <= '0;
                        mem_read_complete <= 1'b0;
                        db_info_loaded <= 1'b0;
                        done <= 1'b0;
                        
                        // Use configured top-k if valid, otherwise use default
                        if (config_top_k > 0 && config_top_k <= TOP_K) begin
                            actual_top_k <= config_top_k;
                        end else begin
                            actual_top_k <= TOP_K;
                        end
                    end
                end
                
                R_LOAD_DB_INFO: begin
                    if (!db_info_loaded) begin
                        if (!mem_rd_en) begin
                            // Start reading database info
                            mem_rd_en <= 1'b1;
                            mem_rd_addr <= 32'h0; // Assume DB info is at address 0
                            mem_words_to_read <= 32'd4; // Need 4 words for DB info
                            mem_words_read <= '0;
                        end else if (mem_rd_valid) begin
                            // Process received data
                            case (mem_words_read)
                                32'd0: db_size <= mem_rd_data[31:0];
                                32'd1: db_addr_start <= mem_rd_data[31:0];
                                32'd2: vector_stride <= mem_rd_data[31:0];
                                32'd3: metadata_addr_start <= mem_rd_data[31:0];
                            endcase
                            
                            mem_words_read <= mem_words_read + 1;
                            mem_rd_addr <= mem_rd_addr + (BUS_WIDTH/8);
                            
                            if (mem_words_read == mem_words_to_read - 1) begin
                                mem_rd_en <= 1'b0;
                                db_info_loaded <= 1'b1;
                            end
                        end
                    end
                end
                
                R_FETCH_VECTOR: begin
                    // Check cache first
                    if (cache_hit) begin
                        // Use cached vector
                        current_vector <= cached_vector;
                        current_doc_idx <= cached_doc_idx;
                        mem_read_complete <= 1'b1;
                    end else if (!mem_read_complete) begin
                        if (!mem_rd_en) begin
                            // Start reading vector
                            mem_rd_en <= 1'b1;
                            mem_rd_addr <= db_addr_start + (vector_idx * vector_stride);
                            mem_words_to_read <= (EMBEDDING_DIM * 4 + BUS_WIDTH - 1) / BUS_WIDTH;
                            mem_words_read <= '0;
                            current_doc_idx <= vector_idx; // Default document index
                        end else if (mem_rd_valid) begin
                            // Process received data
                            for (int i = 0; i < (BUS_WIDTH/32); i++) begin
                                int idx = (mem_words_read * (BUS_WIDTH/32)) + i;
                                if (idx < EMBEDDING_DIM) begin
                                    current_vector[idx] <= mem_rd_data[i*32 +: 32];
                                end
                            end
                            
                            mem_words_read <= mem_words_read + 1;
                            mem_rd_addr <= mem_rd_addr + (BUS_WIDTH/8);
                            
                            if (mem_words_read == mem_words_to_read - 1) begin
                                mem_rd_en <= 1'b0;
                                mem_read_complete <= 1'b1;
                            end
                        end
                    end
                end
                
                R_COMPUTE_SIMILARITY: begin
                    if (!sim_done && !sim_start) begin
                        sim_start <= 1'b1;
                    end
                end
                
                R_UPDATE_TOPK: begin
                    if (!pq_done && !pq_push) begin
                        pq_push <= 1'b1;
                        pq_push_similarity <= similarity_score;
                        pq_push_index <= current_doc_idx;
                    end
                end
                
                R_LOAD_RESULTS: begin
                    // No additional action needed - results are already in outputs
                end
                
                R_FINISHED: begin
                    done <= 1'b1;
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            R_IDLE: begin
                if (start) begin
                    next_state = R_LOAD_DB_INFO;
                end
            end
            
            R_LOAD_DB_INFO: begin
                if (db_info_loaded) begin
                    next_state = R_FETCH_VECTOR;
                end
            end
            
            R_FETCH_VECTOR: begin
                if (cache_hit || mem_read_complete) begin
                    next_state = R_COMPUTE_SIMILARITY;
                end
            end
            
            R_COMPUTE_SIMILARITY: begin
                if (sim_done) begin
                    next_state = R_UPDATE_TOPK;
                end
            end
            
            R_UPDATE_TOPK: begin
                if (pq_done) begin
                    vector_idx = vector_idx + 1;
                    
                    if (vector_idx >= db_size) begin
                        next_state = R_LOAD_RESULTS;
                    end else begin
                        next_state = R_FETCH_VECTOR;
                        mem_read_complete = 1'b0;
                    end
                end
            end
            
            R_LOAD_RESULTS: begin
                next_state = R_FINISHED;
            end
            
            R_FINISHED: begin
                next_state = R_IDLE;
            end
            
            default: next_state = R_IDLE;
        endcase
    end

endmodule
