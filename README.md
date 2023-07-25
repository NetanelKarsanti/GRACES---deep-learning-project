# Deep learning project
## GRACES Algorithm  
#### introduction:
GRACES is a feature selection algorithm in a supervised learning task. It iteratively identifies the best-chosen feature set that results in the largest discount in optimization loss.
The GRACES framework was compiled during the final project of the Deep Learning course. We made adjustments to the original algorithm to adapt it to handle additional learning tasks, especially dealing with tabular data derived from hyperspectral images.
We will present the order of the files in this project : 
```bash
├───Deep-learning-project 
│   ├───Classification-GRACES
│   │   ├───GRACES.py
│   │   ├───reel_test.py
├───Regression-GRACES
│   ├───GRACES.py
│   ├───real_test_reg.py
├─── pandas2matlab.py
├───ref
```
#### Algorithm Testing Instructions
With the adjustments we've made, you can now thoroughly test the algorithm using two study tasks: classification and regression. Each study task has its dedicated folder for convenience and organization.
\\ **Note** :Inside the reel_test and real_test_reg files, you will find a save parameter set to true. This parameter serves the purpose of saving the model performance in the joblib file. Alternatively, if you decide not to save the model performance, you can still view the results in real time by running the appropriate test file (reel_test.py or real_test_reg.py)
