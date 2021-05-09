# **CUDA-based Genetic Algorithm on TravelingSalesman Problem**

## **Instructions to run**

- To clean all the executables, objects and compile fresh.

```
make clean
make all
```
- The TSP Files from ['tsplib'](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/) are stored in the directory
```
/TSPLIB/
```

- To run the Genetic algorithm on GPU for a file x.tsp, you need to run
```
./ga_gpu x.tsp p p1 g g1 m m1
```
- Where p1 is the population of each generation, g1 is the maximum number of generations and m1 is the number of mutations
- Similary, to run the algorithm on CPU, you need to run
```
./ga_cpu x.tsp p p1 g g1 m m1
```
- Run the notebook (plots.ipynb) to get comparision plots for comparision on execution time, solution cost, no of generations for convergence between GPU and CPU.