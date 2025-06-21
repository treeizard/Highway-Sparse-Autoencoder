# 1. Training Base Model For Interpretability
Following guide explains the process for interpreting a simple Sparse Autoencoder

```
cd SparseAutoencoder
```

## Different Methods to ensure consistency:
1. Fix Neuron Permutation with Orthogonality Constraints
2. Anchoring the Neurons

# 2. Notes on Superposition
Superposition occurs when multiple semantic features (input concepts) share overlapping hidden directions in a neural network (in this case our base model). 

We can demonstrate this "overlapping" by computing the inference and polysemanticity between the rows of the MLP. 

- Why did we choose the first layer? No particular reason other than the fact that it is easier to demonstrate the effect of superposition on MLP. 

## 2.1. Interpreting the Two Graphs
### 2.1.1. Polysemanticity Graph
- A high polysemanticity score means the feature of our target is entangled with other features (from the perspective of Neural Network). While a lower score generally imply the feature is monosemantic. 

- Each input feature is encoded not in isolation but through shared hidden neurons. In this case 25 input variables uses shared neurons to represent more complex features. If most features have moderate or high polysemanticity, it means the MLP is not monosemantic.

### 2.1.2. Interference Graph
- Red cells indicate for the two features have strong superposition. Blue cells indicate the features are mostly orthognal.

- For our case, the 25 input features exhibit non-zero pairwise interference — meaning their normalized weight vectors into the first hidden layer are not orthogonal. Instead, many pairs of features share overlapping directions in the 256-dimensional hidden space.

## 2.2. Steps to Demonstrate Superposition
1. Extract Weight Matrix of the first layer of the base model:
$$
W\in R^{d_{hid}\times d_{input}}
$$

2. Normalise to ensure each feature's vector lies on the unit hypersphere. Feature vector here refers to the neuron weights associated with each input variable.
$$
\hat{W}_f = \frac{W_f}{||W_f||_2}
$$

3. Computer the interference Between Features:
$$
I_{f,g} = \hat{W}^{f} \cdot \hat{W}^{g} = \sum_{h=1}^{d_{\text{hid}}} \hat{W}^{f}_{h} \hat{W}^{g}_{h}
$$

4. Calculate the polysemanticity of features:
$$
P_f = || \text{interference}_f || = \sqrt{ \sum_{g \ne f} I_{f,g}^2 }
$$