# Selected Problem: Real-Time Transaction Risk Scoring Engine

## Problem Details

### 1. **Real-Time Transaction Risk Scoring Engine**
- **Problem Type**: Classification with Regression scoring
- **Problem Description**: Develop a system that assigns risk scores (0-100) to cryptocurrency transactions in real-time, integrating traditional AML indicators with blockchain-specific risk factors including wallet clustering, transaction graph analysis, and counterparty reputation scoring.
- **Difficulty Level**: High - requires sub-second processing of complex multi-dimensional data
- **Data Landscape**: Transaction metadata, wallet addresses, transaction amounts, temporal patterns, counterparty databases, sanctions lists, blockchain graph data, gas fees, transaction frequency patterns
- **Recommended Datasets**:
  - **Elliptic Data Set** (Kaggle, Score: 95/100) - Bitcoin transaction graph with 203,769 nodes, labeled illicit/licit patterns
    - URL: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
    - Features: 166 features (94 local + 72 aggregated), timestep, class labels (illicit/licit), transaction IDs
    - Labels: Yes - Binary classification (illicit/licit), ~2% illicit, ~21% licit, ~77% unknown
  - **IBM AML Dataset** (Kaggle, Score: 92/100) - Comprehensive synthetic AML data with risk scoring features
    - URL: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
    - Features: Transaction amounts, sender/receiver IDs, timestamps, transaction types, risk scores, money laundering indicators
    - Labels: Yes - Multi-class with SAR flags, money laundering typologies, and risk classifications
  - **BitcoinHeist Dataset** (UCI, Score: 85/100) - 2.9M Bitcoin transactions with ransomware patterns for risk assessment
    - URL: https://archive.ics.uci.edu/ml/datasets/BitcoinHeistRansomwareAddressDataset
    - Features: Address, year, day, length, weight, count, looped, neighbors, income, labels (ransomware families)
    - Labels: Yes - Multi-class classification with ransomware family types (29 different families + white addresses)
- **Solution Brainstorm**:
  - Descriptive analytics: Transaction volume analysis, address clustering statistics, temporal pattern identification
  - Diagnostic analytics: Root cause analysis of high-risk score assignments, false positive investigation
  - Predictive modeling: Gradient boosting classifiers, neural networks for risk score prediction
  - Prescriptive analytics: Automated transaction blocking rules, dynamic risk threshold adjustment
  - ML algorithms: XGBoost, Random Forest, Graph Neural Networks (GCN, GAT), ensemble methods
  - Advanced techniques: Graph Attention Networks with ResNet architecture (GAT-ResNet), real-time feature engineering
  - Real-time processing: Apache Kafka streaming, Redis caching, low-latency model serving
  - Hybrid approach: Rule-based system for immediate flagging combined with ML refinement

## Model Architecture Options for Classification + Regression Combination

### 1. Sequential Pipeline Architecture
- **Stage 1**: Binary classifier determines if transaction is suspicious/non-suspicious
- **Stage 2**: Regression model assigns risk score (0-100) only to suspicious transactions
- **Benefits**: Computationally efficient, clear decision boundary
- **Implementation**: `if classifier.predict(transaction) == 'suspicious': risk_score = regressor.predict(transaction)`

### 2. Parallel Ensemble Architecture
- **Classification branch**: Predicts illicit/licit probability
- **Regression branch**: Predicts continuous risk score
- **Fusion layer**: Combines outputs using weighted average or learned weights
- **Final output**: `risk_score = α * classification_prob * 100 + β * regression_score`

### 3. Multi-task Learning Architecture
- **Single Neural Network with Dual Outputs**
- **Shared layers**: Dense(128, activation='relu')
- **Classification head**: Dense(1, activation='sigmoid') for binary classification
- **Regression head**: Dense(1, activation='linear') for risk score 0-100
- **Loss function**: `total_loss = classification_loss + λ * regression_loss`

### 4. Hierarchical Risk Scoring
- **Tier 1**: Multi-class classifier (Low/Medium/High risk categories)
- **Tier 2**: Separate regression models for each risk tier
- **Output**: Precise scores within validated risk ranges (Low: 0-30, Medium: 31-70, High: 71-100)

### 5. Ensemble Voting System
- **Model 1**: XGBoost classifier → binary decision
- **Model 2**: Random Forest regressor → continuous score
- **Model 3**: Graph Neural Network → relationship-based score
- **Final decision**: Weighted voting or stacking ensemble

### Recommended: Sequential Pipeline for Real-Time Processing
```python
def score_transaction(transaction_features):
    # Stage 1: Fast binary classification
    is_suspicious = binary_classifier.predict_proba(transaction_features)[1]
    
    if is_suspicious > threshold:
        # Stage 2: Detailed risk scoring
        risk_score = regression_model.predict(transaction_features)[0]
        return min(max(risk_score, 0), 100)  # Clamp to 0-100
    else:
        return is_suspicious * 30  # Low base score for non-suspicious
```

**Advantages for high-frequency transactions:**
- **Performance**: Only suspicious transactions need expensive regression
- **Interpretability**: Clear two-stage decision process
- **Scalability**: Can process high transaction volumes
- **Flexibility**: Different algorithms for each stage (rule-based + ML)

---

*Selected from problem-refinement-technical-research.md on: 2025-08-29*