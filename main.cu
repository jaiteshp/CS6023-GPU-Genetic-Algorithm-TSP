#include <bits/stdc++.h>
#include "TSPLIB_parser.h"
#include <cuda.h>
#include <curand.h>
#include <ctime>

using namespace std;
using std::cout; using std::endl;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

#define dbg cout << __FILE__ << ":" << __LINE__ << ", " << endl
// #define DBL_MAX 1.7976931348623158e+307

const int POP_SIZE = 40000;
const int NUM_GEN = 40;
int NUM_MUTATIONS = 50;
int m = POP_SIZE;
int n;
double **d_cost1, **d_cost2;
double **cost, **d_cost;
double **ccost;
double *X, *Y, *d_X, *d_Y;
int *defaultArr;
int **initialPopulation;
int **pop1, **pop2, **ofsp;
float *rndm;
int RNDM_NUM_COUNT;
bool *shouldStop;
double *bestSolution;

void allocateCudaMemory() {
    // double **temp = (double**) malloc(sizeof(double*)*n);
    double **temp = new double*[n];
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
    
    cudaMallocManaged(&shouldStop, sizeof(bool));
    cudaMallocManaged(&bestSolution, sizeof(double));
    return;
}

__global__ void printCostRow(int n, double *row) {
    int id = (blockIdx.x*blockDim.x)+threadIdx.x;
    if(id > 0)
        return;
    for(int i = 0; i < n; i++) {
        printf("%lf\t", row[i]);
    }
    printf("\n");
    return;
}

void makeInitialPopulation() {
    cudaMalloc(&rndm, sizeof(float)*RNDM_NUM_COUNT);
    int **cpop1, **cpop2, **cofsp;
    ccost = (double **) malloc(sizeof(double*)*n);
    cpop1 = new int*[POP_SIZE];
    cpop2 = new int*[POP_SIZE];
    cofsp = new int*[POP_SIZE];
    cudaMallocManaged(&initialPopulation, sizeof(int*)*POP_SIZE);
    cudaMallocManaged(&d_cost2, sizeof(double*)*n);
    cudaMalloc(&d_cost1, sizeof(double*)*n);
    cudaMalloc(&pop1, sizeof(int*)*POP_SIZE);
    cudaMalloc(&pop2, sizeof(int*)*POP_SIZE);
    cudaMalloc(&ofsp, sizeof(int*)*POP_SIZE);
    for(int i = 0; i < n; i++) {
        cudaMalloc(&ccost[i], sizeof(double)*n);
        cudaMemcpy(&ccost[i], cost[i], sizeof(double)*n, cudaMemcpyHostToDevice);
        cudaMallocManaged(&d_cost2[i], sizeof(double)*n);
    }
    for(int i = 0; i < POP_SIZE; i++) {
        cudaMallocManaged(&initialPopulation[i], sizeof(int)*n);
        cudaMalloc(&cpop1[i], sizeof(int)*n);
        cudaMalloc(&cpop2[i], sizeof(int)*n);
        cudaMalloc(&cofsp[i], sizeof(int)*n);
        random_shuffle(defaultArr, defaultArr+n);
        for(int j = 0; j < n; j++) initialPopulation[i][j] = defaultArr[j];
    }
    cudaMemcpy(d_cost1, ccost, sizeof(double*)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(pop1, cpop1, sizeof(int*)*POP_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(pop2, cpop2, sizeof(int*)*POP_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(ofsp, cofsp, sizeof(int*)*POP_SIZE, cudaMemcpyHostToDevice);
    return;    
}

void initializeBestSolution() {
    *(bestSolution) = 0.0;
    for(int i = 1; i < n; i++) {
        int a = defaultArr[i-1];
        int b = defaultArr[i];
        *(bestSolution) += cost[a][b];
    }
    *(bestSolution) += cost[defaultArr[n-1]][defaultArr[0]];
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

__device__ double computeFitness(int n, int **pop1, int row, double **cost) {
    double pathLength = 0.0;
    for(int i = 1; i < n; i++) {
        int u = pop1[row][i-1];
        int v = pop1[row][i];
        pathLength = pathLength + cost[u][v];
    }
    pathLength += cost[pop1[row][n-1]][pop1[row][0]];
    return pathLength;
}

__device__ int argMaxFitness(int n, int **pop1, int low, int high, double **cost) {
    int idx = 0;
    double mn = 1.7976931348623158e+40;
    for(int row = low; row < high; row++) {
        double fitness = computeFitness(n, pop1, row, cost);
        if(fitness < mn) {
            mn = fitness;
            idx = row;
        }
    }
    return idx;
}

__device__ int getAvlblIdx(int &idx, int n, int a, int b) {
    int res;
    if(idx < a || idx > b) {
        res = idx;
        idx++;
    } else {
        res = b+1;
        idx = b+2;
    }
    return res;
}

__device__ void mutateOffspring(int id, int n, int NUM_MUTATIONS, int **pop2, float *rndm, double **cost) {
    int offset = id*(6+2*(NUM_MUTATIONS))+6;
    for(int mut = 0; mut < NUM_MUTATIONS; mut++) {
        double oldfitness = computeFitness(n, pop2, id, cost);
        int a, b;
        a = n*rndm[offset++];
        b = n*rndm[offset++];

        int temp = pop2[id][a];
        pop2[id][a] = pop2[id][b];
        pop2[id][b] = temp;

        double newFitness = computeFitness(n, pop2, id, cost);

        if(newFitness > oldfitness) {
            int temp = pop2[id][a];
            pop2[id][a] = pop2[id][b];
            pop2[id][b] = temp;
        }
    }
    return;
}

__device__ void adjustRangeOrder(int &a, int &b) {
    if(a > b) {
        int temp = a;
        a = b;
        b = temp;
    } else if(a == b) {
        if(a == 0) b++;
        else a--;
    }
    return ;
}

__device__ bool hasConverged(int n, int POP_SIZE, int **pop2, double **cost) {
    double fitness = computeFitness(n, pop2, 0, cost);
    for(int i = 1; i < POP_SIZE; i++) {
        if((long) fitness != (long) computeFitness(n, pop2, i, cost)) {
            // printf("218, failed at %d, %ld, %ld\n", i, (long) fitness, (long) computeFitness(n, pop2, i, cost));
            return false;
        }
    }
    return true;
}

__global__ void processKernel(int n, int POP_SIZE, int NUM_MUTATIONS, int **pop1, int **pop2, int **pres, double **cost, double *X, double *Y, float *rndm) {
    int id = (blockIdx.x*blockDim.x)+threadIdx.x;
    if(id >= POP_SIZE) 
        return;
    
    int parent1, parent2, low1, high1, low2, high2, a, b;
    int offset = id*(6+2*(NUM_MUTATIONS));
    for(int i = 0; i < 4; i++) 
        rndm[offset+i] = POP_SIZE*rndm[offset+i];
    low1 = rndm[offset+0];
    high1 = rndm[offset+1];
    low2 = rndm[offset+2];
    high2 = rndm[offset+3];
    adjustRangeOrder(low1, high1);
    adjustRangeOrder(low2, high2);

    //////////////////////////////////
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            double temp = cost[i][j];
            temp = temp+1.0;
            cost[i][j] = temp-1.0;
        }
    }
    /////////////////////////////////////

    parent1 = argMaxFitness(n, pop1, low1, high1, cost);
    parent2 = argMaxFitness(n, pop1, low2, high2, cost);
    

    a = n*rndm[offset+4];
    b = n*rndm[offset+5];
    adjustRangeOrder(a, b);

    for(int i = 0; i < n; i++) 
        pres[id][i] = 0;

   
    for(int i = a; i <= b; i++) {
        pop2[id][i] = pop1[parent1][i];
        pres[id][pop1[parent1][i]] = 1;
    }

    
    int avlblIdx = 0;
    for(int i = 0; i < n; i++) {
        int numToInsert = pop1[parent2][i];
        if(pres[id][numToInsert] == 0) {
            pres[id][numToInsert] = 1;
            pop2[id][getAvlblIdx(avlblIdx, n, a, b)] = numToInsert;
        }
    }


    mutateOffspring(id, n, NUM_MUTATIONS, pop2, rndm, cost);

    // if(id % 100 == 0 && id < 1000) 
    //     printf("%dth\t allele with solution: %lf\n", id, computeFitness(n, pop2, id, cost));
    return;    
}

__device__ void updateBestSolution(int n, int POP_SIZE, int **pop2, double **cost, double *bestSolution) {
    double currSolution;
    for(int i = 0; i < POP_SIZE; i++) {
        currSolution = computeFitness(n, pop2, i, cost);
        if(currSolution < *(bestSolution)) {
            *(bestSolution) = currSolution;
        }
    }
    return;
}

__global__ void terminationKernel(int n, int POP_SIZE, int **pop2, double **cost, bool *shouldStop, double *bestSolution) {
    int id = (blockIdx.x*blockDim.x)+threadIdx.x;
    if(id > 0)
        return;
    *(shouldStop) = hasConverged(n, POP_SIZE, pop2, cost);
    updateBestSolution(n, POP_SIZE, pop2, cost, bestSolution);
    printf("Solution: %lf\n", *(bestSolution));
    return;
}

void generateRandomNumbers() {
    curandGenerator_t gen;    
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);      
    auto millisec_since_epoch = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    curandSetPseudoRandomGeneratorSeed(gen, (unsigned int) millisec_since_epoch);  
    // cout << "Time " << millisec_since_epoch << endl;
    curandGenerateUniform(gen, rndm, RNDM_NUM_COUNT);   
    curandDestroyGenerator(gen);    
    cudaDeviceSynchronize();  
    
}

__global__ void printCost(int n, double **cost) {
    int id = (blockIdx.x*blockDim.x)+threadIdx.x;
    if(id > 0) return;
    printf("237, %d\n", n);
    int i, j;
    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            printf("%lf\t", cost[i][j]);
            // printf("yo\t");
        }
        printf("\n");
    }
    printf("hi\n");
    return;
}

__global__ void copyD_cost2ToD_cost1(int n, double **cost1, double **cost2) {
    int id = (blockIdx.x*blockDim.x)+threadIdx.x;
    if(id > 0)
        return;
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            cost1[i][j] = cost2[i][j];
        }
    }
    return;
}

void runGA() {
    for(int genNum = 0; genNum < NUM_GEN; genNum++) {
        cout << "-------------- " << genNum << " --------------" << endl;
        if(genNum == 0) 
            copyKernel<<<ceil(POP_SIZE/(float) 1024), 1024>>>(n, POP_SIZE, pop1, initialPopulation);        
        else 
            copyKernel<<<ceil(POP_SIZE/(float) 1024), 1024>>>(n, POP_SIZE, pop1, pop2);
        cudaDeviceSynchronize();

        generateRandomNumbers();

        processKernel<<<ceil(POP_SIZE/(float) 1024), 1024>>>(n, POP_SIZE, NUM_MUTATIONS, pop1, pop2, ofsp, d_cost1, d_X, d_Y, rndm);
        cudaDeviceSynchronize();

        terminationKernel<<<1, 1>>>(n, POP_SIZE, pop2, d_cost1, shouldStop, bestSolution);
        cudaDeviceSynchronize();

        if(*(shouldStop)) {
            cout << endl << "GA converged in " << genNum+1 << "th generation " << "with best Solution: " << *(bestSolution) << endl;
            return;
        }
    }
    cout << "GA didn't converge. Best solution found is: " << *(bestSolution) << endl;
    return;
}

void printCPUCost() {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            cout << (double) cost[i][j] << "\t";
        }
        cout << endl;
    }
    return;
}

void transposeCosts() {
    for(int i = 0; i < n; i++) {
        for(int j = n-1; j > i; j--) {
            cost[i][j] = cost[j][i];
        }
    }
    return;
}

void copyCostsTod_cost2(){
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            d_cost2[i][j] = cost[i][j];
        }
    }
}

void initializeDefaultArray() {
    defaultArr = new int[n];
    for(int i = 0; i < n; i++) 
        defaultArr[i] = i;
    return;
}

void printHyperParmeters() {
    cout << endl;
    cout << "Number of cities: \t" << n << endl;
    cout << "Population size: \t" << POP_SIZE << endl;
    cout << "Number of mutations: \t" << NUM_MUTATIONS << endl;
    cout << "Max no of generations: \t" << NUM_GEN << endl;
}

int main(int argc, char **argv) {
    string filename = "TSPLIB/";
    filename = filename + argv[1];
    ReadFile(filename, n, cost, X, Y);

    RNDM_NUM_COUNT = POP_SIZE*(6 + 2*NUM_MUTATIONS);

    transposeCosts();

    allocateCudaMemory();

    initializeDefaultArray();
    
    makeInitialPopulation();

    copyCostsTod_cost2();

    copyD_cost2ToD_cost1<<<1,1>>>(n, d_cost1, d_cost2);
    cudaDeviceSynchronize();

    initializeBestSolution();

    auto startTimeGA = chrono::high_resolution_clock::now();
    runGA();
    auto endTimeGA = chrono::high_resolution_clock::now();
    double timeTakenGA = chrono::duration_cast<chrono::nanoseconds>(endTimeGA-startTimeGA).count();
    timeTakenGA = timeTakenGA*(1e-9);
    
    cout << endl << "Execution time (GPU): " << timeTakenGA << " seconds" << endl;
    
    printHyperParmeters();

    return 0;
}
