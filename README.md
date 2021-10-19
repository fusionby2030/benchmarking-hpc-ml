# benchmarking-hpc-ml

Want to check the following:

For varying # of CPU nodes:
- Input dimensionality vs training (fit) times
- Dataset size vs training (fit) times
- performance vs the aboves

Distributed hyperparam search time with dask vs serial for increasing features/dataset size

The models tested are listed below:
- XGBoost
- Pytorch simple (shallow, 4 layer max) FFNN
- Deep model FFNN
- Some benchmarked pytorch model 
