//-----------------------------------------------------------------------------
// File: rag_csd_top.sv
// 
// Description: Top-level module for RAG-CSD FPGA implementation
//              Integrates Embedding, Retrieval, and Augmentation components
//              Configurable for different FPGA platforms and scaling levels
//
// Parameters:
//   EMBEDDING_DIM      - Dimension of embeddings
//   MAX_SEQUENCE_LEN   - Maximum sequence length
//   MAX_TOKENS         - Maximum tokens per query
//   TOP_K              - Number of results to retrieve
//   VECTOR_CACHE_SIZE  - Size of vector cache
//   BUS_WIDTH          - AXI bus width
//   USE_HBM            - Flag to use HBM instead of standard memory
//
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module rag_csd_top #(
    parameter int EMBEDDING_DIM = 384,              // Default for MiniLM-L6
    parameter int MAX_SEQUENCE_LEN = 512,           // Maximum sequence length
    parameter int MAX_TOKENS = 128,                 // Maximum tokens per query
    parameter int TOP_K = 5,                        // Number of results to retrieve
    parameter int VECTOR_CACHE_SIZE = 1024,         // Size of vector cache
    parameter int BUS_WIDTH = 512,                  // AXI bus width
    parameter bit USE_HBM = 0                       // Flag for HBM usage
) (
    // Clock and reset
    input  logic                        clk,
    input  logic                        rst_n,
    
    // AXI-Lite control interface
    input  logic                        s_axil_awvalid,
    output logic                        s_axil_awready,
    input  logic [31:0]                 s_axil_awaddr,
    input  logic                        s_axil_wvalid,
    output logic                        s_axil_wready,
    input  logic [31:0]                 s_axil_wdata,
    output logic                        s_axil_bvalid,
    input  logic                        s_axil_bready,
    output logic [1:0]                  s_axil_bresp,
    input  logic                        s_axil_arvalid,
    output logic                        s_axil_arready,
    input  logic [31:0]                 s_axil_araddr,
    output logic                        s_axil_rvalid,
    input  logic                        s_axil_rready,
    output logic [31:0]                 s_axil_rdata,
    output logic [1:0]                  s_axil_rresp,
    
    // AXI-Stream input interface for queries
    input  logic                        s_axis_query_tvalid,
    output logic                        s_axis_query_tready,
    input  logic [BUS_WIDTH-1:0]        s_axis_query_tdata,
    input  logic                        s_axis_query_tlast,
    
    // AXI-Stream output interface for results
    output logic                        m_axis_result_tvalid,
    input  logic                        m_axis_result_tready,
    output logic [BUS_WIDTH-1:0]        m_axis_result_tdata,
    output logic                        m_axis_result_tlast,
    
    // Memory interface for vector database access
    output logic                        mem_rd_en,
    output logic [31:0]                 mem_rd_addr,
    input  logic [BUS_WIDTH-1:0]        mem_rd_data,
    input  logic                        mem_rd_valid,
    output logic                        mem_wr_en,
    output logic [31:0]                 mem_wr_addr,
    output logic [BUS_WIDTH-1:0]        mem_wr_data,
    input  logic                        mem_wr_ack
);

    // State machine states
    typedef enum logic [3:0] {
        IDLE,
        PARSE_QUERY,
        TOKENIZE,
        ENCODE,
        RETRIEVE,
        COMPUTE_SIMILARITIES,
        SORT_RESULTS,
        AUGMENT,
        OUTPUT_RESULTS
    } rag_state_t;
    
    rag_state_t current_state, next_state;
    
    // Internal signals
    logic encode_start, encode_done;
    logic retrieve_start, retrieve_done;
    logic augment_start, augment_done;
    
    // Control register signals
    logic [31:0] control_reg;            // Control register
    logic [31:0] status_reg;             // Status register
    logic [31:0] top_k_config;           // Configurable top-k value
    logic [1:0]  similarity_metric;      // 0=cosine, 1=dot, 2=euclidean
    logic [31:0] query_len;              // Query length in bytes
    
    // Memory arbiter signals
    logic encoder_mem_rd_en, retrieval_mem_rd_en, augment_mem_rd_en;
    logic [31:0] encoder_mem_rd_addr, retrieval_mem_rd_addr, augment_mem_rd_addr;
    
    // Encoder signals
    logic [EMBEDDING_DIM-1:0][31:0] query_embedding;
    
    // Instantiate encoder module
    encoder #(
        .EMBEDDING_DIM(EMBEDDING_DIM),
        .MAX_TOKENS(MAX_TOKENS),
        .BUS_WIDTH(BUS_WIDTH)
    ) encoder_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(encode_start),
        .done(encode_done),
        .query_data(s_axis_query_tdata),
        .query_valid(s_axis_query_tvalid && s_axis_query_tready),
        .query_last(s_axis_query_tlast),
        .embedding(query_embedding),
        // Memory interface
        .mem_rd_en(encoder_mem_rd_en),
        .mem_rd_addr(encoder_mem_rd_addr),
        .mem_rd_data(mem_rd_data),
        .mem_rd_valid(mem_rd_valid)
    );
    
    // Instantiate retrieval module
    logic [TOP_K-1:0][EMBEDDING_DIM-1:0][31:0] retrieved_vectors;
    logic [TOP_K-1:0][31:0] similarity_scores;
    logic [TOP_K-1:0][31:0] doc_indices;
    
    retrieval #(
        .EMBEDDING_DIM(EMBEDDING_DIM),
        .TOP_K(TOP_K),
        .VECTOR_CACHE_SIZE(VECTOR_CACHE_SIZE),
        .BUS_WIDTH(BUS_WIDTH)
    ) retrieval_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(retrieve_start),
        .done(retrieve_done),
        .similarity_metric(similarity_metric),
        .config_top_k(top_k_config),
        .query_embedding(query_embedding),
        // Memory interface
        .mem_rd_en(retrieval_mem_rd_en),
        .mem_rd_addr(retrieval_mem_rd_addr),
        .mem_rd_data(mem_rd_data),
        .mem_rd_valid(mem_rd_valid),
        // Results
        .retrieved_vectors(retrieved_vectors),
        .similarity_scores(similarity_scores),
        .doc_indices(doc_indices)
    );
    
    // Instantiate augmentation module
    logic [BUS_WIDTH-1:0] augmented_data;
    logic augmented_valid, augmented_last;
    
    augmentation #(
        .MAX_SEQUENCE_LEN(MAX_SEQUENCE_LEN),
        .TOP_K(TOP_K),
        .EMBEDDING_DIM(EMBEDDING_DIM),
        .BUS_WIDTH(BUS_WIDTH)
    ) augmentation_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(augment_start),
        .done(augment_done),
        .query_data(s_axis_query_tdata),
        .retrieved_vectors(retrieved_vectors),
        .similarity_scores(similarity_scores),
        .doc_indices(doc_indices),
        // Memory interface
        .mem_rd_en(augment_mem_rd_en),
        .mem_rd_addr(augment_mem_rd_addr),
        .mem_rd_data(mem_rd_data),
        .mem_rd_valid(mem_rd_valid),
        // Output
        .augmented_data(augmented_data),
        .augmented_valid(augmented_valid),
        .augmented_last(augmented_last)
    );
    
    // AXI-Lite control interface
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axil_awready <= 1'b0;
            s_axil_wready <= 1'b0;
            s_axil_bvalid <= 1'b0;
            s_axil_bresp <= 2'b00;
            s_axil_arready <= 1'b0;
            s_axil_rvalid <= 1'b0;
            s_axil_rdata <= 32'h0;
            s_axil_rresp <= 2'b00;
            
            // Default configuration
            control_reg <= 32'h0;
            status_reg <= 32'h0;
            top_k_config <= TOP_K;
            similarity_metric <= 2'b00; // Default to cosine similarity
        end else begin
            // Write address channel
            if (s_axil_awvalid && !s_axil_awready) begin
                s_axil_awready <= 1'b1;
            end else begin
                s_axil_awready <= 1'b0;
            end
            
            // Write data channel
            if (s_axil_wvalid && !s_axil_wready) begin
                s_axil_wready <= 1'b1;
                
                // Register writes
                case (s_axil_awaddr[7:0])
                    8'h00: control_reg <= s_axil_wdata;
                    8'h04: top_k_config <= s_axil_wdata;
                    8'h08: similarity_metric <= s_axil_wdata[1:0];
                    default: ;
                endcase
            end else begin
                s_axil_wready <= 1'b0;
            end
            
            // Write response channel
            if (s_axil_awready && s_axil_wready && !s_axil_bvalid) begin
                s_axil_bvalid <= 1'b1;
                s_axil_bresp <= 2'b00; // OKAY
            end else if (s_axil_bready && s_axil_bvalid) begin
                s_axil_bvalid <= 1'b0;
            end
            
            // Read address channel
            if (s_axil_arvalid && !s_axil_arready) begin
                s_axil_arready <= 1'b1;
            end else begin
                s_axil_arready <= 1'b0;
            end
            
            // Read data channel
            if (s_axil_arready && !s_axil_rvalid) begin
                s_axil_rvalid <= 1'b1;
                s_axil_rresp <= 2'b00; // OKAY
                
                // Register reads
                case (s_axil_araddr[7:0])
                    8'h00: s_axil_rdata <= control_reg;
                    8'h04: s_axil_rdata <= top_k_config;
                    8'h08: s_axil_rdata <= {30'h0, similarity_metric};
                    8'h0C: s_axil_rdata <= status_reg;
                    default: s_axil_rdata <= 32'h0;
                endcase
            end else if (s_axil_rready && s_axil_rvalid) begin
                s_axil_rvalid <= 1'b0;
            end
            
            // Update status register
            status_reg <= {
                28'h0,
                augment_done,
                retrieve_done,
                encode_done,
                (current_state != IDLE)
            };
        end
    end
    
    // Memory arbiter - controls access to the shared memory interface
    always_comb begin
        // Default: no access
        mem_rd_en = 1'b0;
        mem_rd_addr = 32'h0;
        
        // Priority-based arbitration
        if (encoder_mem_rd_en) begin
            mem_rd_en = encoder_mem_rd_en;
            mem_rd_addr = encoder_mem_rd_addr;
        end else if (retrieval_mem_rd_en) begin
            mem_rd_en = retrieval_mem_rd_en;
            mem_rd_addr = retrieval_mem_rd_addr;
        end else if (augment_mem_rd_en) begin
            mem_rd_en = augment_mem_rd_en;
            mem_rd_addr = augment_mem_rd_addr;
        end
    end
    
    // State machine for controlling the RAG pipeline
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
            encode_start <= 1'b0;
            retrieve_start <= 1'b0;
            augment_start <= 1'b0;
        end else begin
            current_state <= next_state;
            
            // Default values for pulse signals
            encode_start <= 1'b0;
            retrieve_start <= 1'b0;
            augment_start <= 1'b0;
            
            // Generate start pulses on state transitions
            if (current_state == TOKENIZE && next_state == ENCODE) begin
                encode_start <= 1'b1;
            end
            
            if (current_state == ENCODE && next_state == RETRIEVE) begin
                retrieve_start <= 1'b1;
            end
            
            if (current_state == RETRIEVE && next_state == AUGMENT) begin
                augment_start <= 1'b1;
            end
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            IDLE: begin
                if (control_reg[0]) begin // Start bit
                    next_state = PARSE_QUERY;
                end
            end
            
            PARSE_QUERY: begin
                if (s_axis_query_tvalid) begin
                    next_state = TOKENIZE;
                end
            end
            
            TOKENIZE: begin
                next_state = ENCODE;
            end
            
            ENCODE: begin
                if (encode_done) begin
                    next_state = RETRIEVE;
                end
            end
            
            RETRIEVE: begin
                if (retrieve_done) begin
                    next_state = AUGMENT;
                end
            end
            
            AUGMENT: begin
                if (augment_done) begin
                    next_state = OUTPUT_RESULTS;
                end
            end
            
            OUTPUT_RESULTS: begin
                if (augmented_last && augmented_valid && m_axis_result_tready) begin
                    next_state = IDLE;
                end
            end
            
            default: next_state = IDLE;
        endcase
    end
    
    // Output control
    assign s_axis_query_tready = (current_state == PARSE_QUERY) || (current_state == TOKENIZE);
    assign m_axis_result_tvalid = (current_state == OUTPUT_RESULTS) && augmented_valid;
    assign m_axis_result_tdata = augmented_data;
    assign m_axis_result_tlast = augmented_last;

endmodule
