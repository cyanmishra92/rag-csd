//-----------------------------------------------------------------------------
// File: vector_cache.sv
// 
// Description: Vector Cache Module for RAG-CSD
//              Caches frequently accessed vectors to reduce memory traffic
//
// Parameters:
//   EMBEDDING_DIM  - Embedding dimension
//   CACHE_SIZE     - Size of the cache in number of vectors
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module vector_cache #(
    parameter int EMBEDDING_DIM = 384,
    parameter int CACHE_SIZE = 1024
) (
    // Clock and reset
    input  logic                                      clk,
    input  logic                                      rst_n,
    
    // Cache lookup interface
    input  logic [31:0]                               lookup_idx,
    output logic                                      hit,
    output logic [EMBEDDING_DIM-1:0][31:0]            vector,
    output logic [31:0]                               doc_idx,
    
    // Cache store interface
    input  logic                                      store,
    input  logic [31:0]                               store_idx,
    input  logic [EMBEDDING_DIM-1:0][31:0]            store_vector,
    input  logic [31:0]                               store_doc_idx
);

    // Cache entry structure
    typedef struct packed {
        logic valid;                               // Entry valid flag
        logic [31:0] vector_idx;                   // Index of the stored vector
        logic [31:0] doc_idx;                      // Document index
        logic [EMBEDDING_DIM-1:0][31:0] vector;    // Vector data
    } cache_entry_t;
    
    // Cache storage - implement as a direct-mapped cache
    cache_entry_t cache [CACHE_SIZE-1:0];
    
    // Cache index calculation
    function logic [$clog2(CACHE_SIZE)-1:0] calc_cache_idx(input logic [31:0] vector_idx);
        return vector_idx % CACHE_SIZE;
    endfunction
    
    // Cache lookup logic - single cycle response
    always_comb begin
        // Calculate cache index
        logic [$clog2(CACHE_SIZE)-1:0] idx = calc_cache_idx(lookup_idx);
        
        // Check if entry is valid and matches lookup index
        hit = cache[idx].valid && (cache[idx].vector_idx == lookup_idx);
        
        // Output vector and doc_idx if hit
        if (hit) begin
            vector = cache[idx].vector;
            doc_idx = cache[idx].doc_idx;
        end else begin
            vector = '0;
            doc_idx = '0;
        end
    end
    
    // Cache store logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Initialize cache
            for (int i = 0; i < CACHE_SIZE; i++) begin
                cache[i].valid <= 1'b0;
                cache[i].vector_idx <= '0;
                cache[i].doc_idx <= '0;
            end
        end else if (store) begin
            // Store new entry in cache
            logic [$clog2(CACHE_SIZE)-1:0] idx = calc_cache_idx(store_idx);
            
            cache[idx].valid <= 1'b1;
            cache[idx].vector_idx <= store_idx;
            cache[idx].doc_idx <= store_doc_idx;
            cache[idx].vector <= store_vector;
        end
    end

endmodule
