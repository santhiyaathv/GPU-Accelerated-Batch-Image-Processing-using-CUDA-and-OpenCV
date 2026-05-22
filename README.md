# GPU-Accelerated-Batch-Image-Processing-using-CUDA-and-OpenCV

### GPU-Rock-Paper-Scissors-CUDA

### Project Overview
This project demonstrates a simple GPU-based Rock–Paper–Scissors game using CUDA programming. Two GPU players compete against each other by generating moves (Rock, Paper, or Scissors) using CUDA kernels. Each round is executed on the GPU, and the results are transferred to the CPU to determine the winner.

The project helps in understanding CUDA kernels, GPU parallel execution, and device-to-host memory transfer using a fun gaming simulation.

## project Features
```
Features
GPU vs GPU gameplay simulation
CUDA kernel-based move generation
Randomized gameplay using seed logic
Multiple round execution automatically
Winner decision logic implemented in C++
Terminal-based output display
```

### Technologies Used
```
Technologies Used
CUDA Toolkit
C++ Programming Language
NVIDIA GPU
Google Colab / CUDA-enabled environment
Linux Terminal (for execution)
```

### Project Structure
```
Project Structure
GPU-RPS-Battle-CUDA/
│
├── gpu_rps_battle.cu        # Main CUDA program
├── README.md                # Project description
├── output.txt               # Sample output results
├── execution_log.txt        # Run logs (optional)
└── demo_video_link.txt      # Demo video reference
```

### CUDA Implementation of GPU vs GPU Rock Paper Scissors Game Code
```
%%writefile gpu_rps_battle.cu

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

__global__ void generateMoves(int *moves, int seed)
{
    int idx = threadIdx.x;

    // 0 = Rock, 1 = Paper, 2 = Scissors
    moves[idx] = (seed + idx * 13) % 3;
}

const char* name(int move)
{
    if(move == 0) return "Rock";
    if(move == 1) return "Paper";
    return "Scissors";
}

void decideWinner(int p1, int p2)
{
    printf("GPU 1: %s\n", name(p1));
    printf("GPU 2: %s\n", name(p2));

    if(p1 == p2)
    {
        printf("Result: Draw\n");
        return;
    }

    if((p1 == 0 && p2 == 2) ||   // Rock beats Scissors
       (p1 == 1 && p2 == 0) ||   // Paper beats Rock
       (p1 == 2 && p2 == 1))     // Scissors beats Paper
    {
        printf("Winner: GPU 1\n");
    }
    else
    {
        printf("Winner: GPU 2\n");
    }
}

int main()
{
    int *d_moves;
    int h_moves[2];

    cudaMalloc((void**)&d_moves, 2 * sizeof(int));

    srand(time(0));

    for(int round = 1; round <= 5; round++)
    {
        printf("\n===== Round %d =====\n", round);

        int seed = rand();

        generateMoves<<<1, 2>>>(d_moves, seed);

        cudaMemcpy(h_moves, d_moves, 2 * sizeof(int), cudaMemcpyDeviceToHost);

        decideWinner(h_moves[0], h_moves[1]);
    }

    cudaFree(d_moves);

    return 0;
}
```
### OUTPUT

<img width="242" height="441" alt="image" src="https://github.com/user-attachments/assets/67971674-85a2-4830-aeb1-11d5013a7e5c" />

### RESULT
The project successfully demonstrates a GPU vs GPU Rock–Paper–Scissors game using CUDA with automatic move generation, multiple rounds, and winner determination through GPU computation.

