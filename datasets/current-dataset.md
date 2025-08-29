# Selected Dataset: Elliptic Data Set

**Selection Date**: 2025-08-29  
**Selected from**: dataset-exploration-risk-scoring-research.md  
**Rank**: #1 out of 10 evaluated datasets  
**Suitability Score**: 92/100  

## Dataset Overview

### **Elliptic Data Set**
- **Source Platform**: Kaggle
- **Direct URL**: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- **Dataset Size**: 200,000 transactions × 166 features, ~6GB
- **Problem Relevance**: High - Bitcoin illicit transaction classification
- **Data Quality**: Excellent - professionally curated by Elliptic Co.
- **License Type**: Open (with attribution requirements)
- **Last Updated**: 2019 (stable reference dataset)
- **Preprocessing Needs**: Minimal - ready for ML training

## Key Features and Structure

### Feature Categories
- **Local Features**: 94 transaction-specific features
- **Aggregate Features**: 72 neighborhood/graph-based features  
- **Total Features**: 166 feature dimensions
- **Temporal Component**: Time step information included
- **Labels**: Binary classification (illicit/licit)

### Specific Features Include:
- Transaction fees
- Input/output volumes  
- Neighbor aggregates
- Time steps
- BTC amounts
- Graph topology metrics
- Wallet clustering information

## Why This Dataset is Optimal

### **For Classification + Regression Requirements:**
1. **Strong Classification Labels**: Professional-grade illicit/licit labeling
2. **Rich Feature Set**: 166 dimensions provide extensive basis for risk scoring
3. **Graph Features**: Essential for blockchain transaction analysis
4. **Real-time Applicability**: Preprocessed features suitable for sub-second inference
5. **Risk Score Derivation**: Transaction amounts and neighbor aggregates can generate 0-100 risk scores

### **Dataset Statistics:**
- **Class Distribution**: ~2% illicit, ~21% licit, ~77% unknown
- **Scale**: Large enough for robust model training
- **Quality**: Industry-standard dataset used in academic research
- **Stability**: Well-established dataset with consistent structure

## Implementation Strategy

### **Sequential Pipeline Approach (Recommended):**
1. **Stage 1**: Use binary classification labels for illicit/licit detection
2. **Stage 2**: Derive risk scores (0-100) from transaction amounts, graph features, and aggregate metrics

### **Multi-task Learning Alternative:**
- Use classification labels for one output head
- Engineer continuous risk scores from feature combinations for regression head
- Train unified model with dual objectives

### **Feature Engineering for Risk Scoring:**
```python
# Risk score derivation strategy
risk_score = weighted_combination(
    transaction_amount_percentile * 30,
    neighbor_risk_aggregate * 40,
    temporal_anomaly_score * 30
)
# Clamp to 0-100 range
```

## Data Access and Usage

### **Download Information:**
- **Size**: ~6GB compressed
- **Format**: CSV files with node features and edge lists
- **Structure**: Transaction nodes with feature vectors and temporal annotations
- **Requirements**: Kaggle account for download access

### **Preprocessing Steps:**
1. Load transaction features and labels
2. Handle unknown labels (~77% of data) for semi-supervised learning
3. Normalize features for consistent scaling
4. Engineer risk scores from existing feature combinations
5. Split data temporally for realistic evaluation

## Alternative Datasets (Backup Options)

### **Secondary Choice: Bitcoin Transaction Graph Dataset 2025**
- **Score**: 88/100
- **Advantages**: Most recent data (2025), larger scale
- **Disadvantages**: Requires more extensive preprocessing

### **Tertiary Choice: Ethereum Fraud Detection Dataset**  
- **Score**: 85/100
- **Advantages**: Clean Ethereum data, direct fraud labels
- **Disadvantages**: Smaller scale, different blockchain

---

**Selection Rationale**: The Elliptic Data Set provides the optimal balance of data quality, feature richness, and applicability to our Real-Time Transaction Risk Scoring Engine requirements. Its professional curation, extensive feature set, and proven track record in academic research make it the ideal starting point for both classification and risk scoring model development.

*Selected from dataset exploration research conducted on: 2025-08-29*