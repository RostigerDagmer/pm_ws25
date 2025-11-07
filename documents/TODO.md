7.11.2025
TODOs for next 2 week:

  - general: 
      - get access to the gpu/cpu cluster (Kequn)
      - setup and prepare paper writing (not urgent)

  - create an initial process models and traces data set that we can use for testing

  - feature extraction (conceptual)
    - more trace/model features?
    - features correlations
    
  - feature extraction (practical) 
    - create a data structure that combines all features in one, this should hold the values of all features (maybe a trace feature vector and a model feature vector?) 
    - methods to extract different features from a process model and a trace (partialy done)
   
  - generating/evaluating process models (IMPORTANT)
      - Note: the created (dsitributed) variants should be saved somewhere, after discovering/generating the pms
      - because we will have to use those exact variants for the training  
   
  - classifiers would be fitting to solve our problem
      - DecisionTree
      - XGBoosting
      - more?

  - heuristic implemenation
      - add a profiler to calculate only the A* search instead of the whole
      - find a trust worthy number of iterations for alignment computation from a trace and a pm
