# Clustering Approach for Label Estimation
## Elliptic Dataset Semi-Supervised Learning Strategy

**Document Date**: 2025-08-29  
**Target Dataset**: Elliptic Data Set (Kaggle)  
**Objective**: Estimate labels for 154,000 unlabeled transactions using clustering techniques  
**Context**: Real-Time Transaction Risk Scoring Engine preprocessing  

## Problem Overview

The Elliptic Data Set contains:
- **46,000 labeled transactions** (4,000 illicit + 42,000 licit)
- **154,000 unlabeled transactions** (77% of dataset)
- **166 features per transaction** (94 local + 72 aggregate graph features)
- **234,355 directed payment flows** (transaction graph edges)

## Clustering Strategies for Label Estimation

### **1. Graph-Based Clustering**

#### **Approach: Transaction Graph Structure**
```python
# Use the transaction graph structure
from sklearn.cluster import SpectralClustering
from networkx import Graph

# Build transaction graph using the 234,355 edges
G = build_transaction_graph(edges)
# Apply spectral clustering on graph structure
clusters = SpectralClustering(n_clusters=3).fit_predict(graph_features)
```

#### **Implementation Steps:**
1. Construct directed graph from transaction flows
2. Apply spectral clustering to identify graph communities
3. Analyze cluster purity using known labels
4. Assign labels to high-purity clusters

#### **Expected Benefits:**
- Leverages Bitcoin's inherent transaction flow structure
- "Guilt by association" principle for illicit transaction detection
- Natural clustering based on money flow patterns

### **2. Feature-Based K-Means Clustering**

#### **Approach: Multi-Dimensional Feature Space**
```python
# Cluster using the 166 transaction features
from sklearn.cluster import KMeans

# Use known labels to guide cluster interpretation
kmeans = KMeans(n_clusters=5)  # Illicit, Licit, Unknown subgroups
cluster_labels = kmeans.fit_predict(transaction_features)
```

#### **Implementation Steps:**
1. Normalize 166-dimensional feature space
2. Apply K-means with multiple cluster counts (3, 5, 7)
3. Evaluate cluster purity using labeled subset
4. Select optimal cluster count and assign labels

#### **Feature Groups for Clustering:**
- **Local Features (94)**: Transaction-specific patterns
- **Aggregate Features (72)**: Neighborhood behavior patterns
- **Combined**: Full 166-dimensional space

### **3. Semi-Supervised Label Propagation**

#### **Approach: Probabilistic Label Spreading**
```python
from sklearn.semi_supervised import LabelPropagation

# Use 46K labeled + 154K unlabeled transactions
label_propagation = LabelPropagation()
estimated_labels = label_propagation.fit(features, partial_labels)
```

#### **Implementation Steps:**
1. Create label matrix with known labels and -1 for unknown
2. Apply label propagation algorithm
3. Extract probability scores for confidence assessment
4. Assign labels based on confidence thresholds

#### **Advantages:**
- Designed specifically for semi-supervised scenarios
- Provides confidence scores for label assignments
- Handles class imbalance naturally

### **4. Graph-Based Label Propagation**

#### **Approach: Network-Aware Label Spreading**
```python
# Propagate labels through transaction graph
def graph_label_propagation(graph, known_labels, iterations=10):
    for _ in range(iterations):
        for node in unlabeled_nodes:
            neighbor_votes = collect_neighbor_labels(node, graph)
            if confidence(neighbor_votes) > threshold:
                assign_label(node, majority_vote(neighbor_votes))
```

#### **Implementation Logic:**
- "Guilt by association": Connected illicit transactions likely illicit
- "Legitimacy by association": Connected licit transactions likely licit
- Iterative propagation with confidence thresholds

### **5. Anomaly-Based Clustering**

#### **Approach: Outlier Detection for Illicit Identification**
```python
# Identify outliers that might be illicit
from sklearn.ensemble import IsolationForest

isolation_forest = IsolationForest(contamination=0.1)
anomaly_scores = isolation_forest.fit_predict(features)
# High anomaly scores → potential illicit transactions
```

#### **Strategy:**
1. Train anomaly detector on licit transactions
2. Score all unlabeled transactions
3. High anomaly scores → candidate illicit labels
4. Low anomaly scores → candidate licit labels

### **6. Temporal Clustering**

#### **Approach: Time-Series Pattern Analysis**
```python
# Use the 49 time steps for temporal pattern analysis
# Group transactions by temporal behavior patterns
temporal_clusters = cluster_by_time_patterns(transactions, time_steps)
```

#### **Temporal Features:**
- Transaction timing patterns
- Activity bursts vs. steady patterns
- Seasonal behaviors
- Cross-time step correlations

## Validation Strategy

### **1. Conservative Approach**
- Only assign labels to transactions with **high confidence** (>90% similarity to known clusters)
- Keep uncertain transactions unlabeled for safety
- Prioritize precision over recall

### **2. Confidence Scoring Framework**
```python
def estimate_label_confidence(transaction, cluster_center, known_labels):
    feature_distance = calculate_feature_distance(transaction, cluster_center)
    neighbor_label_consensus = check_neighbor_labels(transaction)
    temporal_consistency = analyze_temporal_patterns(transaction)
    
    confidence = weighted_average(feature_distance, consensus, temporal_consistency)
    return confidence
```

### **3. Cross-Validation Protocol**
- Use known 46K labels to validate clustering accuracy
- Test different clustering parameters on labeled subset first
- Measure cluster purity and separation metrics

## Implementation Phases

### **Phase 1: Exploratory Clustering (Validation)**
1. Apply multiple clustering algorithms on **labeled subset only**
2. Measure clustering performance: silhouette score, purity, ARI
3. Identify optimal clustering method and parameters
4. Analyze cluster compositions using known labels

### **Phase 2: Full Dataset Clustering**
1. Apply best clustering method to **full 200K transactions**
2. Identify high-purity clusters (>80% single class)
3. Calculate confidence scores for each unlabeled transaction
4. Create candidate label assignments with confidence levels

### **Phase 3: Label Assignment**
1. **High Confidence** (>90%): Assign definitive labels
2. **Medium Confidence** (70-90%): Assign with uncertainty flag
3. **Low Confidence** (<70%): Keep unlabeled

### **Phase 4: Model Training Enhancement**
1. Train baseline model on original **46K labels**
2. Train enhanced model with **additional estimated labels**
3. Compare performance improvements using cross-validation
4. Iterate clustering parameters based on model performance

## Expected Results

### **Potential Label Expansion:**
- **Conservative Estimate**: ~30,000 additional labels (20% of unlabeled)
  - ~25,000 licit (following 21% vs 2% ratio)
  - ~5,000 illicit
- **Moderate Estimate**: ~50,000 additional labels (32% of unlabeled)
- **Aggressive Estimate**: ~100,000 additional labels (65% of unlabeled)

### **Performance Impact:**
- **Training Data**: 2-3x increase in labeled examples
- **Class Balance**: Improved representation for minority class (illicit)
- **Model Accuracy**: Expected 5-15% improvement in classification performance
- **Risk Scoring**: More robust regression targets for 0-100 risk scoring

## Risk Mitigation

### **Quality Control:**
1. **Validation Set**: Hold out portion of known labels for clustering validation
2. **Human Review**: Manual verification of high-confidence illicit predictions
3. **Ensemble Validation**: Use multiple clustering methods for consensus
4. **Iterative Refinement**: Continuously improve clustering based on model feedback

### **Error Handling:**
- Track false positive/negative rates during validation
- Implement conservative thresholds to minimize mislabeling
- Maintain original labels as ground truth reference

This clustering approach would effectively **double or triple** your labeled training data, providing substantial improvement for both the classification and regression components of your Real-Time Transaction Risk Scoring Engine while maintaining label quality through rigorous validation.

---

*Preprocessing Strategy Document*  
*Created for zzaia-mvp-kyt-pos project on 2025-08-29*