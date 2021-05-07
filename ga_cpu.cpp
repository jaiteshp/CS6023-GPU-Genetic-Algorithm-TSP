#include <bits/stdc++.h>
#include "TSPLIB_parser.h"
#include <ctime>

using namespace std;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

#define dbg cout << __FILE__ << ":" << __LINE__ << ", " << endl

int POP_SIZE = 4000;
int NUM_GEN = 40;
int NUM_MUTATIONS = 50;
int n;
double **cost;
double *X, *Y;
int RNDM_NUM_COUNT;
int *defaultArr;
int **pop1, **pop2, **pres;
bool shouldStop;
double bestSolution;
int **initialPopulation;
float *rndm;

void transposeCosts() {
    for(int i = 0; i < n; i++) {
        for(int j = n-1; j > i; j--) {
            cost[i][j] = cost[j][i];
        }
    }
    return;
}

void initializeDefaultArray() {
    defaultArr = new int[n];
    for(int i = 0; i < n; i++) 
        defaultArr[i] = i;
    return;
}

void makeInitialPopulation() {
    rndm = (float *) malloc(sizeof(float)*RNDM_NUM_COUNT);
    pop1 = (int **) malloc(sizeof(int*)*POP_SIZE);
    pop2 = (int **) malloc(sizeof(int*)*POP_SIZE);
    pres = (int **) malloc(sizeof(int*)*POP_SIZE);

    initialPopulation = (int **) malloc(sizeof(int*)*POP_SIZE);
    for(int i = 0; i < POP_SIZE; i++) {
        pop1[i] = (int *) malloc(sizeof(int)*n);
        pop2[i] = (int *) malloc(sizeof(int)*n);
        pres[i] = (int *) malloc(sizeof(int)*n);
        initialPopulation[i] = (int *) malloc(sizeof(int)*n);
        for(int j = 0; j < n; j++) {
            initialPopulation[i][j] = defaultArr[j];
        }
    }
    return;
}

void initializeBestSolution() {
    bestSolution = 0.0;
    for(int i = 1; i < n; i++) {
        int a = defaultArr[i-1];
        int b = defaultArr[i];
        bestSolution += cost[a][b];
    }
    bestSolution += cost[defaultArr[n-1]][defaultArr[0]];
    return;
}

void printHyperParmeters() {
    cout << endl;
    cout << "Number of cities: \t" << n << endl;
    cout << "Population size: \t" << POP_SIZE << endl;
    cout << "Number of mutations: \t" << NUM_MUTATIONS << endl;
    cout << "Max no of generations: \t" << NUM_GEN << endl;
    return;
}

void copyKernel(int n, int POP_SIZE, int **pop1, int **pop2) {
    for(int id = 0; id < POP_SIZE; id++) {
        for(int i = 0; i < n; i++) {
            pop1[id][i] = pop2[id][i];
        }
    }
    return;
}

float getRandomFloat() {
    float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    return r;
}

void generateRandomNumbers() {   
    auto millisec_since_epoch = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    srand((unsigned int) millisec_since_epoch);
    for(int i = 0; i < RNDM_NUM_COUNT; i++) {
        rndm[i] = getRandomFloat();
    }
    return;
}

void adjustRangeOrder(int &a, int &b) {
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

double computeFitness(int **pop1, int row) {
    double pathLength = 0.0;
    for(int i = 1; i < n; i++) {
        int u = pop1[row][i-1];
        int v = pop1[row][i];
        pathLength = pathLength + cost[u][v];
    }
    pathLength += cost[pop1[row][n-1]][pop1[row][0]];
    return pathLength;
}

int argMaxFitness(int n, int **pop1, int low, int high) {
    int idx = 0;
    double mn = 1.7976931348623158e+40;
    for(int row = low; row < high; row++) {
        double fitness = computeFitness(pop1, row);
        if(fitness < mn) {
            mn = fitness;
            idx = row;
        }
    }
    return idx;
}

int getAvlblIdx(int &idx, int n, int a, int b) {
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

void mutateOffspring(int id, int **pop2) {
    int offset = id*(6+2*(NUM_MUTATIONS))+6;
    for(int mut = 0; mut < NUM_MUTATIONS; mut++) {
        double oldfitness = computeFitness(pop2, id);
        int a, b;
        a = n*rndm[offset++];
        b = n*rndm[offset++];

        int temp = pop2[id][a];
        pop2[id][a] = pop2[id][b];
        pop2[id][b] = temp;

        double newFitness = computeFitness(pop2, id);

        if(newFitness > oldfitness) {
            int temp = pop2[id][a];
            pop2[id][a] = pop2[id][b];
            pop2[id][b] = temp;
        }
    }
    return;
}

void processKernel() {
    for(int id = 0; id < POP_SIZE; id++) {
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

        parent1 = argMaxFitness(n, pop1, low1, high1);
        parent2 = argMaxFitness(n, pop1, low2, high2);

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

        mutateOffspring(id, pop2);
        // cout << "215, " << computeFitness(pop2, id) << endl;
    }
    return;
}

bool hasConverged() {
    double fitness = computeFitness(pop2, 0);
    for(int i = 1; i < POP_SIZE; i++) {
        if((long) fitness != (long) computeFitness(pop2, i)) {
            // printf("218, failed at %d, %ld, %ld\n", i, (long) fitness, (long) computeFitness(n, pop2, i, cost));
            return false;
        }
    }
    return true;
}

void updateBestSolution() {
    double currSolution;
    for(int i = 0; i < POP_SIZE; i++) {
        currSolution = computeFitness(pop2, i);
        if(currSolution < bestSolution) {
            bestSolution = currSolution;
        }
    }
    return;
}

void terminationKernel() {
    shouldStop = hasConverged();
    updateBestSolution();
    printf("Solution: %lf\n", bestSolution);
    return;
}

void runGA() {
    for(int genNum = 0; genNum < NUM_GEN; genNum++) {
        cout << "-------------- " << genNum << " --------------" << endl;
        if(genNum == 0) 
            copyKernel(n, POP_SIZE, pop1, initialPopulation);        
        else 
            copyKernel(n, POP_SIZE, pop1, pop2);

        generateRandomNumbers();

        processKernel();

        terminationKernel();

        if(shouldStop) {
            cout << endl << "GA converged in " << genNum+1 << "th generation " << "with best Solution: " << bestSolution << endl;
            return;
        }
    }
    cout << "GA didn't converge. Best solution found is: " << bestSolution << endl;
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

void takeCmndLineArgs(int argc, char **argv) {
    int numExtraArgc = (argc - 2)/2;
    int argIdx = 2;
    while(numExtraArgc) {
        char c = argv[argIdx++][0];
        int val = atoi(argv[argIdx++]);
        if(c == 'p') {
            POP_SIZE = val;
        } else if(c == 'g') {
            NUM_GEN = val;
        } else if(c == 'm') {
            NUM_MUTATIONS = val;
        }
        numExtraArgc--;
    }
    return;
}

int main(int argc, char **argv) {
    string filename = "TSPLIB/";
    filename = filename + argv[1];
    ReadFile(filename, n, cost, X, Y);

    takeCmndLineArgs(argc, argv);

    RNDM_NUM_COUNT = POP_SIZE*(6 + 2*NUM_MUTATIONS);

    transposeCosts();

    initializeDefaultArray();
    
    makeInitialPopulation();

    initializeBestSolution();

    auto startTimeGA = chrono::high_resolution_clock::now();
    runGA();
    auto endTimeGA = chrono::high_resolution_clock::now();
    double timeTakenGA = chrono::duration_cast<chrono::nanoseconds>(endTimeGA-startTimeGA).count();
    timeTakenGA = timeTakenGA*(1e-9);
    
    cout << endl << "Execution time (CPU): " << timeTakenGA << " seconds" << endl;
    
    printHyperParmeters();

    return 0;
}