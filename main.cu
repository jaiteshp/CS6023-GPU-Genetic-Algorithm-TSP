#include <bits/stdc++.h>
#include "TSPLIB_parser.h"
#include <cuda.h>

using namespace std;
#define dbg cout << __FILE__ << ":" << __LINE__ << ", " << endl

const int POP_SIZE = 100;
const int NUM_GEN = 1;
int m = POP_SIZE;
int n;
double **cost, **d_cost;
double *X, *Y, *d_X, *d_Y;
int *defaultArr;
int **initialPopulation;
int **pop1, **pop2, **ofsp;


void allocateCudaMemory() {
    double **temp = (double**) malloc(sizeof(double*)*n);
    for(int i = 0; i < n; i++) {
        cudaMalloc(&temp[i], sizeof(double)*n);
        cudaMemcpy(&temp[i], cost[i], sizeof(double)*n, cudaMemcpyHostToDevice);
    }

    cudaMalloc(&d_cost, sizeof(double*)*n);
    cudaMemcpy(&d_cost, temp, sizeof(double*)*n, cudaMemcpyHostToDevice);

    cudaMalloc(&d_X, sizeof(double)*n);
    cudaMemcpy(&d_X, X, sizeof(double)*n, cudaMemcpyHostToDevice);

    cudaMalloc(&d_Y, sizeof(double)*n);
    cudaMemcpy(&d_Y, Y, sizeof(double)*n, cudaMemcpyHostToDevice);
    
    return;
}

void makeInitialPopulation() {
    int **cpop1, **cpop2, **cofsp;
    cpop1 = new int*[POP_SIZE];
    cpop2 = new int*[POP_SIZE];
    cofsp = new int*[POP_SIZE];
    cudaMallocManaged(&initialPopulation, sizeof(int*)*POP_SIZE);
    cudaMalloc(&pop1, sizeof(int*)*POP_SIZE);
    cudaMalloc(&pop2, sizeof(int*)*POP_SIZE);
    cudaMalloc(&ofsp, sizeof(int*)*POP_SIZE);
    for(int i = 0; i < POP_SIZE; i++) {
        cudaMallocManaged(&initialPopulation[i], sizeof(int)*n);
        cudaMalloc(&cpop1[i], sizeof(int)*n);
        cudaMalloc(&cpop2[i], sizeof(int)*n);
        cudaMalloc(&cofsp[i], sizeof(int)*n);
        random_shuffle(defaultArr, defaultArr+n);
        for(int j = 0; j < n; j++) initialPopulation[i][j] = defaultArr[j];
    }
    // cudaMemcpy(initialPopulation, cinitialPopulation, sizeof(int*)*POP_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(pop1, cpop1, sizeof(int*)*POP_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(pop2, cpop2, sizeof(int*)*POP_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(ofsp, cofsp, sizeof(int*)*POP_SIZE, cudaMemcpyHostToDevice);
    // for(int i = 0; i < POP_SIZE; i++) {
    //     for(int j = 0; j < n; j++) {
    //         cout << initialPopulation[i][j] << ",";
    //     }
    //     cout << endl;
    // }
    return;    
}

__global__ void copyKernel(int n, int POP_SIZE, int **pop1, int **pop2) {
    int id = (blockIdx.x*blockDim.x)+threadIdx.x;
    if(id >= POP_SIZE) 
        return;
    
    for(int i = 0; i < n; i++) 
        pop1[id][i] = pop2[id][i];

    return;
}

__global__ void processKernel(int n, int POP_SIZE, int **pop1, int **pop2) {
    
    return;
}

void runGA() {
    for(int genNum = 0; genNum < NUM_GEN; genNum++) {
        if(genNum == 0) 
            copyKernel<<<ceil(POP_SIZE/(float) 1024), 1024>>>(n, POP_SIZE, pop1, initialPopulation);        
        else 
            copyKernel<<<ceil(POP_SIZE/(float) 1024), 1024>>>(n, POP_SIZE, pop1, pop2);
        cudaDeviceSynchronize();
    }
    return;
}

int main(int argc, char **argv) {
    string filename = "TSPLIB/";
    filename = filename + argv[1];
    cout << filename << endl;
    ReadFile(filename, n, cost, X, Y);

    allocateCudaMemory();

    defaultArr = new int[n];
    for(int i = 0; i < n; i++) 
        defaultArr[i] = i;
    
    makeInitialPopulation();

    runGA();
}
