struct KernelConfig
{
    const int BlockRowWarps;
    const int BlockColWarps;
    const int WarpRowTiles;
    const int WarpColTiles;
    const int PatchM;
    const int PatchN;
    const int ChunkK;
    const int NumStages;
    const int PipelineStrategy;
    const int kWMMA_M;
    const int kWMMA_N;
    const int kWMMA_K;
    const int K;
    const int N;
};

constexpr KernelConfig quadmul_4096_57344_8192 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 2,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 2,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 8192,
    /* N */ 57344};

constexpr KernelConfig quadmul_4096_8192_8192 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 2,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 2,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 8192,
    /* N */ 8192};

constexpr KernelConfig quadmul_4096_28672_4096 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 2,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 2,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 4096,
    /* N */ 28672};

constexpr KernelConfig quadmul_4096_10240_8192 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 2,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 2,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 8192,
    /* N */ 10240};

constexpr KernelConfig quadmul_4096_6144_4096 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 2,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 2,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 4096,
    /* N */ 6144};

constexpr KernelConfig quadmul_2048_8192_28672 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 28672,
    /* N */ 8192};

constexpr KernelConfig quadmul_2048_10240_8192 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 8192,
    /* N */ 10240};

constexpr KernelConfig quadmul_2048_8192_8192 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 8192,
    /* N */ 8192};

constexpr KernelConfig quadmul_2048_28672_4096 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 4096,
    /* N */ 28672};

constexpr KernelConfig quadmul_256_10240_8192 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 8192,
    /* N */ 10240};

constexpr KernelConfig quadmul_256_4096_14336 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 2,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 14336,
    /* N */ 4096};

constexpr KernelConfig quadmul_1024_6144_4096 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 4096,
    /* N */ 6144};

constexpr KernelConfig quadmul_256_6144_4096 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 2,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 4096,
    /* N */ 6144};

constexpr KernelConfig quadmul_256_28672_4096 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 4096,
    /* N */ 28672};

constexpr KernelConfig quadmul_128_10240_8192 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 2,
    /* WarpColTiles */ 4,
    /* PatchM */ 1,
    /* PatchN */ 2,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 8192,
    /* N */ 10240};

constexpr KernelConfig quadmul_256_57344_8192 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 8192,
    /* N */ 57344};

constexpr KernelConfig quadmul_1024_28672_4096 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 4096,
    /* N */ 28672};

constexpr KernelConfig quadmul_1024_4096_14336 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 2,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 2,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 14336,
    /* N */ 4096};

constexpr KernelConfig quadmul_128_8192_8192 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 2,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 1,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 8192,
    /* N */ 8192};

constexpr KernelConfig quadmul_4096_4096_14336 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 14336,
    /* N */ 4096};

constexpr KernelConfig quadmul_128_28672_4096 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 4096,
    /* N */ 28672};

constexpr KernelConfig quadmul_512_6144_4096 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 4096,
    /* N */ 6144};

constexpr KernelConfig quadmul_128_57344_8192 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 2,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 8192,
    /* N */ 57344};

constexpr KernelConfig quadmul_512_8192_28672 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 2,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 2,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 28672,
    /* N */ 8192};

constexpr KernelConfig quadmul_1024_10240_8192 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 8192,
    /* N */ 10240};

constexpr KernelConfig quadmul_128_8192_28672 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 2,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 1,
    /* PatchN */ 2,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 28672,
    /* N */ 8192};

constexpr KernelConfig quadmul_128_6144_4096 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 2,
    /* WarpColTiles */ 4,
    /* PatchM */ 2,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 4096,
    /* N */ 6144};

constexpr KernelConfig quadmul_2048_6144_4096 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 4096,
    /* N */ 6144};

constexpr KernelConfig quadmul_2048_4096_14336 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 14336,
    /* N */ 4096};

constexpr KernelConfig quadmul_512_57344_8192 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 8192,
    /* N */ 57344};

constexpr KernelConfig quadmul_128_4096_14336 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 2,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 1,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 14336,
    /* N */ 4096};

constexpr KernelConfig quadmul_256_8192_28672 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 2,
    /* PatchN */ 2,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 28672,
    /* N */ 8192};

constexpr KernelConfig quadmul_1024_8192_28672 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 2,
    /* PatchN */ 2,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 28672,
    /* N */ 8192};

constexpr KernelConfig quadmul_512_28672_4096 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 4096,
    /* N */ 28672};

constexpr KernelConfig quadmul_4096_8192_28672 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 2,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 2,
    /* PatchN */ 2,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 28672,
    /* N */ 8192};

constexpr KernelConfig quadmul_512_4096_14336 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 2,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 2,
    /* PatchN */ 2,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 14336,
    /* N */ 4096};

constexpr KernelConfig quadmul_512_8192_8192 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 2,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 2,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 8192,
    /* N */ 8192};

constexpr KernelConfig quadmul_512_10240_8192 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 2,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 2,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 8192,
    /* N */ 10240};

constexpr KernelConfig quadmul_1024_8192_8192 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 1,
    /* PatchN */ 4,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 8192,
    /* N */ 8192};

constexpr KernelConfig quadmul_2048_57344_8192 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 8192,
    /* N */ 57344};

constexpr KernelConfig quadmul_256_8192_8192 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 2,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 2,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 8192,
    /* N */ 8192};

constexpr KernelConfig quadmul_1024_57344_8192 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* PatchM */ 4,
    /* PatchN */ 1,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* kWMMA_M */ 8,
    /* kWMMA_N */ 8,
    /* kWMMA_K */ 32,
    /* K */ 8192,
    /* N */ 57344};
