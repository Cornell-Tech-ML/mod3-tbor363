# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

- Docs: https://minitorch.github.io/

- Overview: https://minitorch.github.io/module3.html

You will need to modify `tensor_functions.py` slightly in this assignment.

- Tests:

```
python run_tests.py
```

- Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

# TASK 3.4 matrix multiply cpu vs gpu performance time

![plot](training_pics/3.4.png)

# TASK 3.5 model training

# CPU TRAINING

# Simple Dataset

![plot](training_pics/cpu_simple1.png)

![plot](training_pics/cpu_simple2.png)

# XOR Dataset

![plot](training_pics/cpu_xor1.png)

![plot](training_pics/cpu_xor2.png)

# Split Dataset

![plot](training_pics/cpu_split1.png)

![plot](training_pics/cpu_split2.png)

# GPU TRAINING

# Simple Dataset

![plot](training_pics/gpu_simple1.png)

![plot](training_pics/gpu_simple2.png)

# XOR Dataset

![plot](training_pics/gpu_xor1.png)

![plot](training_pics/gpu_xor2.png)

# Split Dataset

![plot](training_pics/gpu_split1.png)

![plot](training_pics/gpu_split2.png)
