# Federated Graph Learning Baselines

This document presents the performance comparison between different federated graph learning approaches under varying levels of client heterogeneity (controlled by parameter β).

## Performance Results

### Performance on Small Datasets

### Cora Dataset (10 clients)

| Method | Centralized | β=1 | β=100 | β=10000 |
|--------|-------------|-----|-------|---------|
| **Centralized GCN** | 0.8069±0.0065 | - | - | - |
| **FedGCN(0-hop)** | - | 0.6502±0.0127 | 0.5958±0.0176 | 0.5992±0.0226 |
| **BDS-GCN** | - | 0.7598±0.0143 | 0.7467±0.0117 | 0.7479±0.018 |
| **FedSage+** | - | 0.8026±0.0054 | 0.7942±0.0075 | 0.796±0.0075 |
| **FedGCN(1-hop)** | - | 0.81±0.0066 | 0.8009±0.007 | 0.8009±0.0077 |
| **FedGCN(2-hop)** | - | 0.8064±0.0043 | 0.8084±0.0051 | 0.8087±0.0061 |

### Citeseer Dataset (10 clients)

| Method | Centralized | β=1 | β=100 | β=10000 |
|--------|-------------|-----|-------|---------|
| **Centralized GCN** | 0.6914±0.0051 | - | - | - |
| **FedGCN(0-hop)** | - | 0.617±0.0118 | 0.5841±0.0168 | 0.5841±0.0138 |
| **BDS-GCN** | - | 0.6709±0.0184 | 0.6596±0.0128 | 0.6582±0.01 |
| **FedSage+** | - | 0.6977±0.0097 | 0.6856±0.0121 | 0.688±0.0086 |
| **FedGCN(1-hop)** | - | 0.7006±0.0071 | 0.6891±0.0067 | 0.693±0.0069 |
| **FedGCN(2-hop)** | - | 0.6933±0.0067 | 0.6953±0.0069 | 0.6948±0.0032 |

### Performance on Large Datasets

### Ogbn-Arxiv Dataset (10 clients)

| Method | Centralized | β=1 | β=100 | β=10000 |
|--------|-------------|-----|-------|---------|
| **Centralized GCN** | 0.7±0.0082 | - | - | - |
| **FedGCN(0-hop)** | - | 0.5981±0.0094 | 0.5809±0.0017 | 0.5804±0.0015 |
| **BDS-GCN** | - | 0.6769±0.0086 | 0.6689±0.0024 | 0.6688±0.0015 |
| **FedSage+** | - | 0.7053±0.0073 | 0.6921±0.0014 | 0.6918±0.0024 |
| **FedGCN(1-hop)** | - | 0.7101±0.0078 | 0.6989±0.0038 | 0.7004±0.0031 |
| **FedGCN(2-hop)** | - | 0.712±0.0075 | 0.6972±0.0075 | 0.7017±0.0081 |

### Ogbn-Products Dataset (5 clients)

| Method | Centralized | β=1 | β=100 | β=10000 |
|--------|-------------|-----|-------|---------|
| **Centralized GCN** | 0.7058±0.0008 | - | - | - |
| **FedGCN(0-hop)** | - | 0.6789±0.0031 | 0.658±0.0008 | 0.658±0.0008 |
| **BDS-GCN** | - | 0.6996±0.0019 | 0.6952±0.0012 | 0.6952±0.0009 |
| **FedSage+** | - | 0.7044±0.0017 | 0.7047±0.0009 | 0.7051±0.0006 |
| **FedGCN(1-hop)** | - | 0.7049±0.0016 | 0.7057±0.0003 | 0.7057±0.0004 |
| **FedGCN(2-hop)** | - | 0.7053±0.002 | 0.7057±0.0009 | 0.7055±0.0006 |

### FedGAT Results

The table below presents the performance of FedGAT and other methods on three datasets: Cora, Citeseer, and Pubmed. The results are reported for both IID and non-IID settings with 10 clients.

| Method                  | Cora          | Citeseer      | Pubmed        |
|-------------------------|---------------|---------------|---------------|
| **GCN**                | 0.805         | 0.672         | 0.758         |
| **GAT**                | 0.813         | 0.725         | 0.795         |
| **DistGAT (10 clients, non-IID)** | 0.684 ± 0.013 | 0.664 ± 0.014 | 0.769 ± 0.01  |
| **DistGAT (10 clients, IID)**     | 0.645 ± 0.012 | 0.635 ± 0.007 | 0.745 ± 0.0092|
| **FedGCN (10 clients, non-IID)**  | 0.778 ± 0.003 | 0.68 ± 0.002  | 0.772 ± 0.003 |
| **FedGCN (10 clients, IID)**      | 0.771 ± 0.003 | 0.682 ± 0.002 | 0.774 ± 0.002 |
| **FedGAT (10 clients, non-IID)**  | 0.80 ± 0.005  | 0.699 ± 0.008 | 0.789 ± 0.004 |
| **FedGAT (10 clients, IID)**      | 0.802 ± 0.003 | 0.694 ± 0.006 | 0.787 ± 0.005 |

## Key Observations

- FedGCN with higher hop numbers (1-hop, 2-hop) generally performs better than 0-hop variants across all datasets
- Performance degradation with increasing heterogeneity (lower β values) is less severe for methods with neighborhood communication
- On large datasets like Ogbn-Products, the performance gap between different methods becomes smaller

## Model Architectures and Hyperparameters

### FedGCN Model Architecture and Hyperparameters

#### Main Model Architectures

**Cora and Citeseer Datasets:**
- Architecture: 2-layer GCN
- Hidden units: 16
- First layer activation: ReLU
- Second layer activation: Softmax
- Dropout rate: 0.5 (between the two GCN layers)
- Normalization: Adjacency matrix normalized by Ã = D^(-1/2)AD^(-1/2)

**OGBN-Arxiv Dataset:**
- Architecture: 3-layer GCN
- Hidden units: 256
- Normalization: BatchNorm1d between GCN layers
- Note: BatchNorm1d was added to ensure consistent performance for deeper models

**OGBN-Products Dataset:**
- Architecture: 2-layer GraphSage
- Hidden units: 256

#### Training Hyperparameters

**Cora and Citeseer:**
- Optimizer: SGD
- Learning rate: 0.5
- L2 regularization: 5×10^(-4)
- Training rounds: 300
- Local steps per round: 3 (for federated settings)

**OGBN-Arxiv:**
- Training rounds: 600
- Hidden units: 256

**OGBN-Products:**
- Training rounds: 450
- Hidden units: 256

#### Ablation Studies

**GCN Depth Study (Figure 2):**
- Varied number of layers from 2 to 10
- Compared centralized GCN vs. FedGCN with L-hop, 2-hop, 1-hop, and 0-hop communication
- Found that 2-hop communication is sufficient for up to 10-layer GCNs
- BatchNorm1d was added for OGBN-Arxiv to ensure consistent performance across depths

**Number of Clients Study (Figure 5):**
- Varied number of clients from 2 to over 180
- Tested on Cora dataset
- Compared FedGCN(0-hop), FedGCN(1-hop), and FedGCN(2-hop)
- Found that in cross-device settings (many clients), 2-hop communication is necessary to maintain high model accuracy

**Data Distribution Study:**
- Used Dirichlet distribution with parameter β
- β = 10000 (i.i.d.)
- β = 100 (partial non-i.i.d.)
- β = 1 (strong non-i.i.d.)

**Homomorphic Encryption Configuration (Table 5):**
- Scheme: Cheon-Kim-Kim-Song (CKKS)
- Ring dimension: 4096
- Security level: HEStd_128_classic
- Multi depth: 1 (configured for optimal minimum possible multiplicative depth)
- Scale factor bits: 30

The experimental settings used the same hyperparameters as in the original papers for each dataset (Kipf and Welling (2016) for Cora/Citeseer and Hu et al. (2020) for OGBN datasets).

### FedGAT Model Architecture and Hyperparameters

#### Graph Attention Network (GAT) Architecture

**Cora Dataset:**
- Architecture: 2-layer GAT
- Hidden dimensions: 8
- Attention heads: 8 in the first layer, 1 in the output layer
- Activation function: ReLU/ELU (implied from text)
- Regularization: 0.001
- Learning rate: 0.1
- Optimizer: Adam

**Citeseer Dataset:**
- Architecture: 2-layer GAT
- Hidden dimensions: 8
- Attention heads: 8 in the first layer, 1 in the output layer
- Same hyperparameters as Cora

**Pubmed Dataset:**
- Architecture: 2-layer GAT
- Hidden dimensions: 8
- Attention heads: 8 in first layer, 8 in output layer (differs from Cora/Citeseer)
- Other hyperparameters same as Cora

#### FedGAT Implementation

- Approximation degree: 16 (Chebyshev polynomial approximation degree)
- Parameter aggregation: FedAvg
- Privacy approach: Feature aggregation across nodes

#### Ablation Studies

**Approximation Degree Study:**
- Tested approximation degrees: 8 to 16
- Study performed on three different data distributions:
  - Non-IID (β = 1)
  - Partial IID (β = 100)
  - IID (β = 10000)

**Client Scaling Study:**
- Number of clients tested: 1 to 20 for standard tests
- Extended tests: 20 to 100 clients for Vector FedGAT variant

**Data Distribution Study:**
- Dirichlet distribution parameter β:
  - β = 1 (non-IID)
  - β = 100 (partially IID)
  - β = 10000 (IID)

**Vector FedGAT Variant:**
- Same model architecture as basic FedGAT
- Different computation approach for reduced communication overhead
- Communicates 1-dimensional vectors instead of matrices
- Achieves approximately 10× communication speed-up

The paper also mentions that the experiments were conducted on an AWS m5.16xlarge instance with 256 GiB memory.

### Original GCN Architecture and Hyperparameters

Based on the paper "Semi-Supervised Classification with Graph Convolutional Networks" by Kipf and Welling (2017):

#### Main GCN Architecture

The GCN model uses a layer-wise propagation rule defined as:

H^(l+1) = σ(D̃^(-1/2)ÃD̃^(-1/2)H^(l)W^(l))

Where:
- Ã = A + I_N (adjacency matrix with added self-connections)
- D̃_ii = ∑_j Ã_ij (degree matrix)
- W^(l) is the layer-specific trainable weight matrix
- σ is an activation function like ReLU

#### Primary Model Configuration

- Depth: 2-layer GCN (shown to be optimal for many tasks)
- Input Layer: Dimension depends on the dataset's feature size
- Hidden Layer: 16 units
- Output Layer: Dimension equals the number of classes in the dataset

#### Dataset-Specific Hyperparameters

**Citation Networks (Citeseer, Cora, Pubmed):**
- Dropout Rate: 0.5
- L2 Regularization: 5×10^(-4)
- Number of Hidden Units: 16
- Learning Rate: 0.01 (Adam optimizer)
- Training Iterations: Up to 200 epochs with early stopping (window size of 10)
- Weight Initialization: Glorot & Bengio (2010)

**NELL Dataset:**
- Dropout Rate: 0.1
- L2 Regularization: 1×10^(-5)
- Number of Hidden Units: 64
- Other parameters: Same as above

#### Model Variants in Ablation Studies

**Propagation Model Variants:**
1. Chebyshev Filter:
   - K=3 polynomial expansion
   - K=2 polynomial expansion
2. 1st-order Model: XΘ_0 + D^(-1/2)AD^(-1/2)XΘ_1
3. Single Parameter Model: (I_N + D^(-1/2)AD^(-1/2))XΘ
4. Renormalization Trick (Standard GCN): D̃^(-1/2)ÃD̃^(-1/2)XΘ
5. 1st-order Term Only: D^(-1/2)AD^(-1/2)XΘ
6. Multi-layer Perceptron: XΘ (no graph structure)

**Depth Study Variants:**
- Models tested: 1 to 10 layers
- Standard GCN: Using regular propagation rule
- Residual GCN: Adding residual connections between hidden layers
- Propagation rule: H^(l+1) = σ(D̃^(-1/2)ÃD̃^(-1/2)H^(l)W^(l)) + H^(l)
- Training: 400 epochs without early stopping
- Other hyperparameters: Same as citation networks

**Karate Club Visualization Study:**
- Depth: 3-layer GCN
- Hidden Layer Size: 4 units (chosen to avoid saturation of tanh units)
- Output Layer: 2 units (for 2D visualization)
- Learning Rate: 0.01
- Training Iterations: 300

#### Performance Notes

- Best performance achieved with 2 or 3 layer models
- Models deeper than 7 layers become difficult to train without residual connections
- The standard GCN with renormalization trick outperformed other propagation models
- This GCN approach was shown to be both more accurate and more efficient than previous methods for semi-supervised node classification on graph-structured data.