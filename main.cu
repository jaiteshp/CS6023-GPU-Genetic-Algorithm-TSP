#include <bits/stdc++.h>
#include "TSPLIB_parser.h"
#include <cuda.h>

using namespace std;
#define dbg //cout << __FILE__ << ":" << __LINE__ << ", " << endl

int n;
double **cost;
double *X, *Y;

int main(int argc, char **argv) {
    string filename = "TSPLIB/";
    dbg;
    filename = filename + argv[1];
    dbg;
    cout << filename << endl;
    ReadFile(filename, n, cost, X, Y);
    dbg;

    for(int i = 0; i < n; i++) {
        cout << i+1 << "\t" << X[i] << "\t" << Y[i] << endl;
    }
    cout << cost[0][1] << "\t" << cost[1][0] << endl;
}
