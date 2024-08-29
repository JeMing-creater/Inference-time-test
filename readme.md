# Readme

## Environment install
### Attention
Mamba is only supported on CUDA 11.6 and above. If existed version of CUDA is not 11.6 and above, please follow the setting below:
```
https://blog.csdn.net/m0_53883779/article/details/135701971
```
Some system would lack the support of VS c++, which can not be supported by VS 2022, user can download the VS 2019 from follow:
```
https://pan.baidu.com/s/1BQHqtshV8zhUJ8vfs-t5PA?pwd=vs19 
```

In addition, there are still some problems for building the mamba_ssm in window system. If there are something wrongs in your environment building, please follow the steps in follow link to revise some code in the setting files:
```
https://blog.csdn.net/yyywxk/article/details/140420538
```

### Install causal-conv1d

```
cd environment/causal-conv1d

python setup.py install
```

### Install mamba

```
cd environment/mamba

python setup.py install
```

### Install monai

```
pip install monai
```

## Inference time test
```
python inference.py
```