# Training Base Model For Interpretability
Following guide explains the process for training a base model for Interpretability Purposes

### 1. Training Base Model
```
python train.py --
```

```
tensorboard --logdir=log/[Your Model Directory]
```

### FAQ:
1. Potential Version Error/Unable to Detect for markupsafe, solve by reinstalling 
```
pip uninstall markupsafe -y
pip install markupsafe==2.1.5
```
