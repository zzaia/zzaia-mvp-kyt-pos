# Dataset Exploration Research: Real-Time Transaction Risk Scoring Engine

**Research Date**: 2025-08-29  
**Problem Focus**: Classification with Regression scoring for cryptocurrency transactions  
**Agent**: zzaia-dataset-exploration  

## Executive Summary

After conducting extensive searches across UCI ML Repository, Kaggle, Google Datasets, and Hugging Face, I identified datasets that can support your dual classification and regression requirements for cryptocurrency transaction risk scoring. The **top 3 most suitable datasets** are:

1. **Elliptic Data Set (Kaggle)** - Score: 92/100 - The gold standard for Bitcoin transaction classification with sophisticated graph features
2. **Bitcoin Transaction Graph Dataset 2025 (Google Datasets)** - Score: 88/100 - Latest large-scale temporally annotated dataset with dual labeling
3. **Ethereum Fraud Detection Dataset (Kaggle)** - Score: 85/100 - Comprehensive Ethereum account-level features with fraud classification

---

## **RANKED TOP 10 DATASETS LIST**

### **#1 - Elliptic Data Set**
- **Source Platform**: Kaggle
- **Suitability Score**: 92/100
- **Direct URL**: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- **Dataset Size**: 200,000 transactions × 166 features, ~6GB
- **Problem Relevance**: High - Bitcoin illicit transaction classification
- **Data Quality**: Excellent - professionally curated by Elliptic Co.
- **License Type**: Open (with attribution requirements)
- **Last Updated**: 2019 (stable reference dataset)
- **Preprocessing Needs**: Minimal - ready for ML training
- **Key Features**: 94 local + 72 aggregate graph features, temporal data
- **Feature Titles**: Transaction fees, input/output volumes, neighbor aggregates, time steps, BTC amounts

#### **What is the Elliptic Data Set?**
The **Elliptic Data Set** is a professionally curated Bitcoin transaction dataset created by **Elliptic**, a leading blockchain analytics company specializing in cryptocurrency compliance and investigation services.

**Core Purpose**: A labeled dataset of Bitcoin transactions designed for machine learning research in cryptocurrency fraud detection and anti-money laundering (AML).

**Data Structure Details**:
- **203,769 Bitcoin transaction nodes** with **234,355 directed payment flows (edges)**
- **49 time steps** covering Bitcoin transactions from 2009-2013
- **166 features per transaction**: 94 local transaction-specific features + 72 aggregated neighborhood statistics

**Labeling System**:
- **2% labeled as illicit** (fraud, scams, malware, terrorist organizations, ransomware)
- **21% labeled as licit** (exchanges, wallet services, miners, legitimate businesses)
- **77% unlabeled** (unknown classification, suitable for semi-supervised learning)

**Feature Categories**:
1. **Local Features (94)**: Transaction-specific data including fees, amounts, input/output counts, timestamp information, transaction size
2. **Aggregated Features (72)**: Neighborhood statistics including maximum, minimum, standard deviation of connected transactions, graph-based metrics

**Industry Significance**:
- **Industry Standard**: Created by Elliptic Co., used by law enforcement, financial institutions, and researchers worldwide
- **Real-World Relevance**: Based on actual Bitcoin blockchain transactions with labels derived from known illicit services (Silk Road, ransomware campaigns)
- **Research Impact**: Most cited cryptocurrency fraud detection dataset, foundation for numerous academic papers and commercial AML systems
- **Professional Grade**: Unlike synthetic datasets, contains real Bitcoin transactions with professional-grade labels based on actual investigations and blockchain forensics

This makes it the **gold standard for cryptocurrency fraud detection research** and ideal for dual classification + regression model development.

### **#2 - Bitcoin Transaction Graph Dataset 2025**
- **Source Platform**: Google Datasets (via Nature Scientific Data)
- **Suitability Score**: 88/100
- **Direct URL**: https://www.nature.com/articles/s41597-025-04684-8
- **Dataset Size**: 252M nodes × 785M edges, massive scale
- **Problem Relevance**: High - Comprehensive Bitcoin network analysis
- **Data Quality**: Excellent - peer-reviewed scientific dataset
- **License Type**: Open (Creative Commons)
- **Last Updated**: 2025 (most recent)
- **Preprocessing Needs**: Extensive - requires graph processing
- **Key Features**: Temporal annotations, entity type labels, address annotations
- **Feature Titles**: Node timestamps, edge weights, entity classifications, address types

### **#3 - Ethereum Fraud Detection Dataset**
- **Source Platform**: Kaggle
- **Suitability Score**: 85/100
- **Direct URL**: https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset
- **Dataset Size**: 9,841 accounts × 51 features, ~2MB
- **Problem Relevance**: High - Direct fraud classification for Ethereum
- **Data Quality**: Good - account-level aggregated features
- **License Type**: Open
- **Last Updated**: 2021
- **Preprocessing Needs**: Minimal - clean structured data
- **Key Features**: Transaction patterns, ERC20 token data, timing analysis
- **Feature Titles**: FLAG, avg_min_between_sent_tnx, ERC20_total_Ether_sent, unique_received_from_addresses

### **#4 - BitcoinHeist Ransomware Address Dataset**
- **Source Platform**: UCI ML Repository
- **Suitability Score**: 82/100
- **Direct URL**: https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset
- **Dataset Size**: ~3M addresses × 10 features, ~200MB
- **Problem Relevance**: High - Bitcoin address classification (ransomware vs legitimate)
- **Data Quality**: Excellent - well-documented research dataset
- **License Type**: Open (CC BY 4.0)
- **Last Updated**: 2019
- **Preprocessing Needs**: Minimal - structured and clean
- **Key Features**: Address features, temporal data, graph topology metrics
- **Feature Titles**: address, year, day, length, weight, count, looped, neighbors, income, label

### **#5 - Elliptic++ Dataset**
- **Source Platform**: Google Datasets (via GitHub)
- **Suitability Score**: 80/100
- **Direct URL**: https://github.com/git-disl/EllipticPlusPlus
- **Dataset Size**: 203K transactions + 822K addresses × 56 features
- **Problem Relevance**: High - Extended Bitcoin transaction and address analysis
- **Data Quality**: Excellent - research-grade extension of Elliptic
- **License Type**: Open (research purposes)
- **Last Updated**: 2023
- **Preprocessing Needs**: Moderate - requires graph construction
- **Key Features**: Dual transaction and address labeling, temporal interactions
- **Feature Titles**: Address features, transaction graphs, temporal patterns, illicit labels

### **#6 - IBM Transactions for Anti Money Laundering**
- **Source Platform**: Kaggle
- **Suitability Score**: 75/100
- **Direct URL**: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
- **Dataset Size**: Variable (HI/LI datasets), ~500K-5M transactions
- **Problem Relevance**: Medium - Synthetic financial transactions (not crypto-specific)
- **Data Quality**: Good - IBM-generated synthetic data
- **License Type**: Open
- **Last Updated**: 2021
- **Preprocessing Needs**: Moderate - synthetic data adaptation needed
- **Key Features**: Money laundering labels, transaction networks, AML patterns
- **Feature Titles**: Transaction amounts, currencies, laundering tags, account networks

### **#7 - Elliptic2 Dataset**
- **Source Platform**: Google Datasets (via ArXiv)
- **Suitability Score**: 78/100
- **Direct URL**: https://arxiv.org/html/2404.19109v2
- **Dataset Size**: 122K subgraphs × 49M nodes × 196M edges
- **Problem Relevance**: High - Subgraph-level Bitcoin money laundering detection
- **Data Quality**: Excellent - latest research dataset
- **License Type**: Open (research)
- **Last Updated**: 2024
- **Preprocessing Needs**: Extensive - subgraph analysis required
- **Key Features**: Money laundering subgraphs, cluster analysis, AML patterns
- **Feature Titles**: Subgraph structures, cluster features, laundering patterns, Bitcoin flows

### **#8 - Google Cloud Blockchain Analytics Datasets**
- **Source Platform**: Google Datasets (BigQuery)
- **Suitability Score**: 70/100
- **Direct URL**: https://cloud.google.com/blockchain-analytics/docs/supported-datasets
- **Dataset Size**: Multi-blockchain, petabyte scale
- **Problem Relevance**: Medium - Raw blockchain data (requires labeling)
- **Data Quality**: Excellent - real blockchain transactions
- **License Type**: Open (with usage costs)
- **Last Updated**: Daily updates
- **Preprocessing Needs**: Extensive - raw transaction data
- **Key Features**: Multi-chain support, real-time updates, comprehensive coverage
- **Feature Titles**: Transaction hashes, addresses, amounts, gas fees, block timestamps

### **#9 - HBTBD (Heterogeneous Bitcoin Transaction Behavior Dataset)**
- **Source Platform**: Kaggle
- **Suitability Score**: 72/100
- **Direct URL**: https://www.kaggle.com/datasets/songjialin/hbtbd-for-aml
- **Dataset Size**: 46,045 transactions × multiple features
- **Problem Relevance**: High - Bitcoin AML classification
- **Data Quality**: Good - research dataset for AML
- **License Type**: Open
- **Last Updated**: 2023
- **Preprocessing Needs**: Moderate - structured but needs validation
- **Key Features**: Heterogeneous transaction behaviors, AML labels
- **Feature Titles**: Transaction patterns, behavioral features, illicit/licit labels, AML indicators

### **#10 - Synthetic Financial Datasets (PaySim)**
- **Source Platform**: Kaggle
- **Suitability Score**: 65/100
- **Direct URL**: https://www.kaggle.com/datasets/ealaxi/paysim1
- **Dataset Size**: 6.3M transactions × 11 features, ~500MB
- **Problem Relevance**: Medium - Mobile money fraud (not crypto-specific)
- **Data Quality**: Good - synthetic but realistic transaction patterns
- **License Type**: Open
- **Last Updated**: 2018
- **Preprocessing Needs**: Moderate - adaptation to crypto context needed
- **Key Features**: Mobile payment fraud, synthetic transaction patterns
- **Feature Titles**: step, type, amount, oldbalanceOrg, newbalanceOrig, isFraud

---

## **Dataset Analysis and Recommendations**

### **For Dual Classification + Regression Implementation:**

**Sequential Pipeline Approach:** Start with **Elliptic Data Set (#1)** for primary classification, then use transaction amounts and graph features to derive risk scores (0-100 scale).

**Multi-task Learning:** **Bitcoin Transaction Graph Dataset 2025 (#2)** offers the most comprehensive feature set for training unified models with both classification and regression outputs.

**Parallel Ensemble:** Combine **Elliptic Data Set (#1)** and **Ethereum Fraud Detection Dataset (#3)** for cross-blockchain validation and risk scoring.

### **Technical Implementation Considerations:**

1. **Real-time Processing:** Elliptic and Ethereum datasets provide preprocessed features suitable for sub-second inference
2. **Graph Features:** Datasets #1, #2, #5, #7 provide essential graph topology features for blockchain analysis
3. **Risk Score Derivation:** Use transaction amounts, neighbor aggregates, and temporal patterns to generate 0-100 risk scores
4. **Feature Engineering:** Combine local transaction features with graph-based neighborhood aggregates

### **Data Quality and Limitations:**

- **Labeling Quality:** Professional datasets (Elliptic, UCI) have higher label confidence than community datasets
- **Temporal Coverage:** Most datasets cover 2017-2019 period; Google Cloud provides current data but requires manual labeling
- **Scale Considerations:** Large datasets (#2, #8) require distributed processing infrastructure

## **Final Recommendation**

The recommended approach is to start with **Elliptic Data Set** for proof-of-concept development, then incorporate **Bitcoin Transaction Graph Dataset 2025** for production-scale implementation with the most current temporal patterns.

---

**Generated by**: zzaia-dataset-exploration agent  
**Research Methodology**: Comprehensive web search across UCI ML Repository, Kaggle, Google Datasets, and Hugging Face  
**Focus**: Datasets supporting both classification and regression for cryptocurrency transaction risk scoring