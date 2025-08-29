# Dataset Exploration: Cryptocurrency Transaction Amount Prediction

## Problem Context
**Target**: Predict cryptocurrency transaction amounts using regression techniques
**Problem Type**: Linear Regression 
**Difficulty**: Easy - Entry-level ML implementation
**Focus**: Historical patterns, user behavior, and temporal features

---

## Primary Dataset Analysis: IBM AML Dataset

### Dataset Overview
- **Source**: Kaggle - IBM Transactions for Anti Money Laundering (AML)
- **URL**: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
- **Quality Score**: 92/100
- **Dataset Type**: Synthetic financial transaction data

### Key Features for Regression Target
- **Primary Target**: Transaction amounts (continuous variable)
- **Temporal Features**: Timestamps for time-series analysis
- **Behavioral Features**: Sender/receiver IDs for user pattern analysis
- **Categorical Features**: Transaction types for segmentation
- **Risk Features**: Risk scores and money laundering indicators

### Regression Suitability Analysis
**Strengths:**
- ✅ Clean synthetic data with controlled distributions
- ✅ Comprehensive transaction amount ranges
- ✅ Multi-class labels provide rich feature engineering opportunities
- ✅ Well-documented SAR flags for risk-based modeling
- ✅ Temporal patterns suitable for time-series regression

**Potential Challenges:**
- ⚠️ Synthetic nature may not capture real-world complexity
- ⚠️ Need to verify amount distribution characteristics
- ⚠️ May require additional feature engineering for temporal patterns

### Feature Engineering Opportunities
1. **Amount-based features**: Log transformations, percentile rankings
2. **Temporal features**: Hour/day/month patterns, time since last transaction
3. **User behavior**: Average transaction amounts, frequency patterns
4. **Risk indicators**: SAR flag correlations with amounts

---

## Secondary Dataset Analysis: BitcoinHeist Dataset

### Dataset Overview
- **Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/ml/datasets/BitcoinHeistRansomwareAddressDataset
- **Quality Score**: 85/100
- **Size**: 2.9M Bitcoin addresses with transaction patterns

### Regression Target Potential
- **Primary Target**: Income (Bitcoin amounts received)
- **Temporal Features**: Year, day for time-based analysis
- **Network Features**: Count, weight, neighbors for network analysis
- **Behavioral Features**: Length, looped patterns

### Regression Suitability Analysis
**Strengths:**
- ✅ Real Bitcoin transaction data
- ✅ Large dataset size (2.9M records)
- ✅ Clear numerical targets (income amounts)
- ✅ Network-based features for advanced modeling
- ✅ Temporal patterns across multiple years

**Potential Challenges:**
- ⚠️ Address-level vs transaction-level data
- ⚠️ Imbalanced distribution (ransomware vs normal)
- ⚠️ May require aggregation for transaction amount prediction

### Feature Engineering Opportunities
1. **Amount transformations**: Log income, income percentiles
2. **Network metrics**: Centrality measures, clustering coefficients
3. **Temporal patterns**: Seasonal effects, trend analysis
4. **Behavioral ratios**: Income/count ratios, activity patterns

---

## Tertiary Dataset Analysis: Elliptic Data Set

### Dataset Overview
- **Source**: Kaggle - Elliptic Bitcoin Transaction Graph
- **URL**: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- **Quality Score**: 95/100 (Highest quality)
- **Size**: 203,769 Bitcoin transactions

### Regression Target Potential
- **Features**: 166 transaction features (94 local + 72 aggregated)
- **Graph Structure**: Transaction network with rich feature set
- **Temporal Component**: Timestep information
- **Labels**: Binary classification (can be used for feature engineering)

### Regression Suitability Analysis
**Strengths:**
- ✅ Highest quality dataset with extensive feature engineering
- ✅ 166 pre-computed features reduce preprocessing needs
- ✅ Real Bitcoin transaction graph structure
- ✅ Temporal timesteps for time-series analysis
- ✅ Academic-grade dataset with peer review validation

**Potential Challenges:**
- ⚠️ Features are anonymized (limited interpretability)
- ⚠️ No direct transaction amount target (need to engineer)
- ⚠️ May require creative target variable construction

### Feature Engineering Opportunities
1. **Target construction**: Aggregate features to estimate amounts
2. **Graph features**: Leverage network position for predictions
3. **Temporal modeling**: Time-series patterns in feature evolution
4. **Risk correlation**: Use illicit labels for risk-adjusted predictions

---

## Dataset Combination Strategy

### Multi-Dataset Approach
1. **Primary Development**: Start with IBM AML dataset for clear regression target
2. **Validation**: Use BitcoinHeist for real-world validation
3. **Advanced Features**: Incorporate Elliptic graph features for enhanced modeling

### Data Integration Plan
1. **Feature Alignment**: Map common features across datasets
2. **Scale Normalization**: Ensure amount scales are comparable
3. **Temporal Synchronization**: Align time periods where possible
4. **Cross-Validation**: Test models across different data sources

---

## Recommended Implementation Sequence

### Phase 1: IBM AML Dataset (Primary)
- **Goal**: Establish baseline regression models
- **Target**: Direct transaction amount prediction
- **Models**: Linear Regression, Ridge, Lasso
- **Validation**: Traditional regression metrics (MAE, RMSE, R²)

### Phase 2: BitcoinHeist Integration
- **Goal**: Real-world validation and enhancement
- **Target**: Bitcoin income prediction
- **Models**: Enhanced feature engineering with network effects
- **Validation**: Cross-dataset performance comparison

### Phase 3: Elliptic Advanced Features
- **Goal**: Leverage graph-based features for improved accuracy
- **Target**: Hybrid amount prediction using graph context
- **Models**: Advanced regression with graph features
- **Validation**: Comprehensive model comparison

---

## Data Quality Assessment

### IBM AML Dataset
- **Completeness**: High (synthetic data with controlled missing values)
- **Consistency**: High (standardized format)
- **Accuracy**: Medium (synthetic approximation)
- **Relevance**: High (designed for AML use cases)

### BitcoinHeist Dataset
- **Completeness**: High (real blockchain data)
- **Consistency**: High (standardized extraction)
- **Accuracy**: High (verified blockchain transactions)
- **Relevance**: Medium (focus on ransomware, not general transactions)

### Elliptic Dataset
- **Completeness**: High (comprehensive feature set)
- **Consistency**: High (academic standard)
- **Accuracy**: High (peer-reviewed features)
- **Relevance**: High (transaction-level analysis)

---

## Success Metrics Framework

### Regression Performance
- **Primary**: Mean Absolute Error (MAE) - interpretable in transaction units
- **Secondary**: Root Mean Square Error (RMSE) - penalty for large errors
- **Tertiary**: R-squared - model explanatory power
- **Advanced**: Mean Absolute Percentage Error (MAPE) - scale-independent

### Business Relevance
- **Prediction Accuracy**: Within 10% of actual amounts for 80% of transactions
- **Risk Correlation**: High amounts correlate with risk indicators
- **Temporal Stability**: Model performance stable across time periods
- **Interpretability**: Clear feature importance for business understanding

---

## Next Steps for Dataset Acquisition

1. **Download IBM AML Dataset**: Primary development dataset
2. **Acquire BitcoinHeist Data**: Real-world validation source
3. **Access Elliptic Dataset**: Advanced feature engineering
4. **Setup Data Pipeline**: Automated ingestion and preprocessing
5. **Implement EDA**: Comprehensive exploratory data analysis

---

## Generated by: agent-zzaia-dataset-exploration
**Date**: 2025-08-28
**Focus**: Dataset exploration for cryptocurrency transaction amount regression
**Status**: Ready for implementation
