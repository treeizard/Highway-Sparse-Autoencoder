# Training Base Model For Interpretability
Following guide explains the process for training a base model to be interpreted.

```
cd BaseModel
```

### 1. Training Base Model
```
python train.py --
```

```
tensorboard --logdir=log/[Your Model Directory]
```

### 2. Test Base Model

Different Name available: 
- highway
- merge

```
python3 test.py all --model super
```

### 3. Convert Model to Torch
Different Name available: 
- highway
- merge
```
python3 convert.py super
```

### FAQ:
1. Potential Version Error/Unable to Detect for markupsafe, solve by reinstalling 
```
pip uninstall markupsafe -y
pip install markupsafe==2.1.5
```
