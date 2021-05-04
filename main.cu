#include <bits/stdc++.h>
#include "TSPLIB_parser.h"
#include <cuda.h>

using namespace std;
#define dbg cout << __FILE__ << ":" << __LINE__ << ", " << endl

int n;
double **cost;
double *X, *Y;

int main(int argc, char **argv) {
    string filename = "TSPLIB/";
    filename = filename + argv[1];
    cout << filename << endl;
    ReadFile(filename, n, cost, X, Y);

}
