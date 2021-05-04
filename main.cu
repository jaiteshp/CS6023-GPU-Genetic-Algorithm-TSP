#include <bits/stdc++.h>
#include "TSPLIB_parser.h"
#include <cuda.h>

using namespace std;
#define dbg cout << __FILE__ << ":" << __LINE__ << ", " << endl

const int POP_SIZE = 100;
int n;
double **cost, **d_cost;
double *X, *Y, *d_X, *d_Y;
int *defaultArr;
int **initialPopulation;


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
    cudaMallocManaged(&initialPopulation, sizeof(int*)*POP_SIZE);
    for(int i = 0; i < POP_SIZE; i++) {
        cudaMallocManaged(&initialPopulation[i], sizeof(int)*n);
        random_shuffle(defaultArr, defaultArr+n);
        for(int j = 0; j < n; j++) initialPopulation[i][j] = defaultArr[j];
    }
    for(int i = 0; i < POP_SIZE; i++) {
        for(int j = 0; j < n; j++) {
            cout << initialPopulation[i][j] << ",";
        }
        cout << endl;
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

    
}
