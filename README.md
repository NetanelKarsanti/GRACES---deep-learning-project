# Deep learning project
## GRACES Algorithm  
#### introduction:
GRACES is a feature selection algorithm in a supervised learning task. It iteratively identifies the best-chosen feature set that results in the largest discount in optimization loss.
The GRACES framework was compiled during the final project of the Deep Learning course. We made adjustments to the original algorithm to adapt it to handle additional learning tasks, especially dealing with tabular data derived from hyperspectral images.
```bash
├───analyze
│   ├───plantations
│   │   ├───train12_test3
│   │   ├───train13_test2
│   │   └───train23_test1
│   └───random
├───best_models
├───best_models_plantations
│   ├───train12_test3
│   ├───train13_test2
│   └───train23_test1
├───channels
│   ├───add_one
│   ├───drop_more
│   ├───drop_one
│   └───random_select
├───channels_plantations
│   ├───train12_test3
│   │   ├───add_one
│   │   ├───drop_more
│   │   ├───drop_one
│   │   └───random_select
│   ├───train13_test2
│   │   ├───add_one
│   │   ├───drop_more
│   │   ├───drop_one
│   │   └───random_select
│   └───train23_test1
│       ├───add_one
│       ├───drop_more
│       ├───drop_one
│       └───random_select
├───graphs
│   ├───best_channel_number
│   └───total_best_wl
├───graphs_plantations
│   ├───train12_test3
│   │   ├───best_channel_number
│   │   └───total_best_wl
│   ├───train13_test2
│   │   ├───best_channel_number
│   │   └───total_best_wl
│   └───train23_test1
│       ├───best_channel_number
│       └───total_best_wl
├───ref
```
