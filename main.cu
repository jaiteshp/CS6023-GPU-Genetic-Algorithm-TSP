#include <bits/stdc++.h>
#include "TSPLIB_parser.h"
#include <cuda.h>

using namespace std;
int n;
double **cost;
double *X, *Y;

int main(int argc, char **argv) {
    string filename;
    filename = argv[1];
    ReadFile(filename, n, cost, X, Y);
}
