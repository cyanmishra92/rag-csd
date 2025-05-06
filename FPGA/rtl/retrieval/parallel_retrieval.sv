//-----------------------------------------------------------------------------
// File: parallel_retrieval.sv
// 
// Description: Parallel Retrieval Module for RAG-CSD
//              Computes similarity with multiple vectors in parallel
//
// Parameters:
//   EMBEDDING_DIM           - Embedding dimension
//   TOP_K                   - Number of results to retrieve
//   VECTOR_CACHE_SIZE       - Size of vector cache
//   NUM_PARALLEL_UNITS      - Number of parallel processing units
//   BUS_WIDTH               - AXI bus width
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module parallel_retrieval #(
    parameter int EMBEDDING_DIM = 384,
    parameter int TOP_K = 5,
    parameter int VECTOR_CACHE_SIZE = 1024,
    parameter int NUM_PARALLEL_UNITS = 4,
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

    // Parallel retrieval states
    typedef enum logic [3:0] {
        PR_IDLE,
        PR_LOAD_DB_INFO,
        PR_LOAD_VECTORS,
        PR_COMPUTE_SIMILARITY,
        PR_COLLECT_RESULTS,
        PR_UPDATE_TOPK,
        PR_FINISHED
    } parallel_state_t;
    
    parallel_state_t current_state, next_state;
    
    // Internal registers
    logic [31:0] db_size;                   // Number of vectors in database
    logic [31:0] current_batch;             // Current batch being processed
    logic [31:0] total_batches;             // Total number of batches
    logic [31:0] db_addr_start;             // Start address of vector database
    logic [31:0] vector_stride;             // Bytes per vector
    logic [31:0] metadata_addr_start;       // Start address of metadata
    
    // Current vector batch
    logic [NUM_PARALLEL_UNITS-1:0][EMBEDDING_DIM-1:0][31:0] current_vectors;
    logic [NUM_PARALLEL_UNITS-1:0][31:0] current_doc_indices;
    
    // Parallel similarity units
    logic [NUM_PARALLEL_UNITS-1:0] sim_start, sim_done;
    logic [NUM_PARALLEL_UNITS-1:0][31:0] similarities;
    
    // Top-K management
    logic pq_push, pq_done;
    logic [31:0] best_similarity;
    logic [31:0] best_doc_idx;
    logic [EMBEDDING_DIM-1:0][31:0] best_vector;
    logic [31:0] best_idx;
    
    // Memory interface state
    logic [31:0] mem_words_to_read;
    logic [31:0] mem_words_read;
    logic db_info_loaded;
    logic batch_loaded;
    
    // Instantiate parallel similarity computers
    genvar i;
    generate
        for (i = 0; i < NUM_PARALLEL_UNITS; i++) begin : sim_units
            similarity_computer #(
                .EMBEDDING_DIM(EMBEDDING_DIM)
            ) sim_comp (
                .clk(clk),
                .rst_n(rst_n),
                .start(sim_start[i]),
                .done(sim_done[i]),
                .metric(similarity_metric),
                .vec_a(query_embedding),
                .vec_b(current_vectors[i]),
                .similarity(similarities[i])
            );
        end
    endgenerate
    
    // Instantiate top-k priority queue
    top_k_priority_queue #(
        .K(TOP_K),
        .EMBEDDING_DIM(EMBEDDING_DIM)
    ) topk_queue (
        .clk(clk),
        .rst_n(rst_n),
        .push(pq_push),
        .done(pq_done),
        .similarity(best_similarity),
        .doc_index(best_doc_idx),
        .vector(best_vector),
        .topk_similarities(similarity_scores),
        .topk_indices(doc_indices),
        .topk_vectors(retrieved_vectors)
    );
    
    // State machine logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= PR_IDLE;
            current_batch <= '0;
            total_batches <= '0;
            db_size <= '0;
            db_addr_start <= '0;
            vector_stride <= '0;
            metadata_addr_start <= '0;
            db_info_loaded <= 1'b0;
            batch_loaded <= 1'b0;
            mem_rd_en <= 1'b0;
            mem_rd_addr <= '0;
            mem_words_to_read <= '0;
            mem_words_read <= '0;
            best_similarity <= '0;
            best_doc_idx <= '0;
            best_idx <= '0;
            
            // Reset control signals
            for (int j = 0; j < NUM_PARALLEL_UNITS; j++) begin
                sim_start[j] <= 1'b0;
            end
            
            pq_push <= 1'b0;
            done <= 1'b0;
        end else begin
            current_state <= next_state;
            
            // Default pulse signals
            for (int j = 0; j < NUM_PARALLEL_UNITS; j++) begin
                sim_start[j] <= 1'b0;
            end
            
            pq_push <= 1'b0;
            
            case (current_state)
                PR_IDLE: begin
                    if (start) begin
                        current_batch <= '0;
                        db_info_loaded <= 1'b0;
                        batch_loaded <= 1'b0;
                        done <= 1'b0;
                    end
                end
                
                PR_LOAD_DB_INFO: begin
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
                                
                                // Calculate total batches
                                total_batches <= (db_size + NUM_PARALLEL_UNITS - 1) / NUM_PARALLEL_UNITS;
                            end
                        end
                    end
                end
                
                PR_LOAD_VECTORS: begin
                    if (!batch_loaded) begin
                        if (!mem_rd_en) begin
                            // Start loading batch of vectors
                            mem_rd_en <= 1'b1;
                            mem_words_to_read <= (EMBEDDING_DIM * 4 * NUM_PARALLEL_UNITS + BUS_WIDTH - 1) / BUS_WIDTH;
                            mem_words_read <= '0;
                            
                            // Calculate base address for this batch
                            mem_rd_addr <= db_addr_start + (current_batch * NUM_PARALLEL_UNITS * vector_stride);
                        end else if (mem_rd_valid) begin
                            // Process received data - distribute to parallel units
                            for (int j = 0; j < (BUS_WIDTH/32); j++) begin
                                int unit_idx = (mem_words_read * (BUS_WIDTH/32) + j) / EMBEDDING_DIM;
                                int embed_idx = (mem_words_read * (BUS_WIDTH/32) + j) % EMBEDDING_DIM;
                                
                                if (unit_idx < NUM_PARALLEL_UNITS && 
                                    current_batch * NUM_PARALLEL_UNITS + unit_idx < db_size) begin
                                    current_vectors[unit_idx][embed_idx] <= mem_rd_data[j*32 +: 32];
                                    current_doc_indices[unit_idx] <= current_batch * NUM_PARALLEL_UNITS + unit_idx;
                                end
                            end
                            
                            mem_words_read <= mem_words_read + 1;
                            mem_rd_addr <= mem_rd_addr + (BUS_WIDTH/8);
                            
                            if (mem_words_read == mem_words_to_read - 1) begin
                                mem_rd_en <= 1'b0;
                                batch_loaded <= 1'b1;
                            end
                        end
                    end
                end
                
                PR_COMPUTE_SIMILARITY: begin
                    // Start similarity computation for all units in parallel
                    for (int j = 0; j < NUM_PARALLEL_UNITS; j++) begin
                        if (current_batch * NUM_PARALLEL_UNITS + j < db_size && !sim_done[j]) begin
                            sim_start[j] <= 1'b1;
                        end
                    end
                end
                
                PR_COLLECT_RESULTS: begin
                    // Find best similarity from parallel units
                    best_similarity <= '0;
                    best_idx <= '0;
                    
                    for (int j = 0; j < NUM_PARALLEL_UNITS; j++) begin
                        if (current_batch * NUM_PARALLEL_UNITS + j < db_size && 
                            similarities[j] > best_similarity) begin
                            best_similarity <= similarities[j];
                            best_doc_idx <= current_doc_indices[j];
                            best_vector <= current_vectors[j];
                            best_idx <= j;
                        end
                    end
                end
                
                PR_UPDATE_TOPK: begin
                    // Push best result to top-k queue
                    if (!pq_done) begin
                        pq_push <= 1'b1;
                    end
                    
                    // Prepare for next batch
                    if (pq_done) begin
                        current_batch <= current_batch + 1;
                        batch_loaded <= 1'b0;
                    end
                end
                
                PR_FINISHED: begin
                    done <= 1'b1;
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            PR_IDLE: begin
                if (start) begin
                    next_state = PR_LOAD_DB_INFO;
                end
            end
            
            PR_LOAD_DB_INFO: begin
                if (db_info_loaded) begin
                    next_state = PR_LOAD_VECTORS;
                end
            end
            
            PR_LOAD_VECTORS: begin
                if (batch_loaded) begin
                    next_state = PR_COMPUTE_SIMILARITY;
                end
            end
            
            PR_COMPUTE_SIMILARITY: begin
                // Check if all active units are done
                logic all_done = 1'b1;
                for (int j = 0; j < NUM_PARALLEL_UNITS; j++) begin
                    if (current_batch * NUM_PARALLEL_UNITS + j < db_size && !sim_done[j]) begin
                        all_done = 1'b0;
                    end
                end
                
                if (all_done) begin
                    next_state = PR_COLLECT_RESULTS;
                end
            end
            
            PR_COLLECT_RESULTS: begin
                next_state = PR_UPDATE_TOPK;
            end
            
            PR_UPDATE_TOPK: begin
                if (pq_done) begin
                    if (current_batch < total_batches - 1) begin
                        next_state = PR_LOAD_VECTORS;
                    end else begin
                        next_state = PR_FINISHED;
                    end
                end
            end
            
            PR_FINISHED: begin
                next_state = PR_IDLE;
            end
            
            default: next_state = PR_IDLE;
        endcase
    end

endmodule
