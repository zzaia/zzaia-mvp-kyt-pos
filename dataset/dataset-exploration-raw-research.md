# Top 10 Datasets List for Cryptocurrency Transaction Pattern Analysis

## Executive Summary
The search identified 12 high-quality datasets across the four specified sources. The top 3 most suitable datasets are: (1) Elliptic Bitcoin Dataset for comprehensive graph-based AML analysis, (2) IBM AML Synthetic Dataset for traditional banking compliance integration, and (3) UCI BitcoinHeist Dataset for ransomware and illicit transaction detection.

---

## **#1 - Elliptic Bitcoin Dataset**
- **Source Platform**: Kaggle
- **Suitability Score**: 95/100
- **Direct URL**: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- **Dataset Size**: 203,769 nodes × 166 features, ~50MB
- **Problem Relevance**: High - Specifically designed for AML compliance
- **Data Quality**: Excellent - Industry-standard reference dataset
- **License Type**: CC BY-NC-ND 4.0 (Restricted commercial use)
- **Last Updated**: 2019 (stable reference dataset)
- **Preprocessing Needs**: Minimal - Ready for graph analysis
- **Key Features**: Bitcoin transaction graph, 166 node features (94 local + 72 aggregated), 2% illicit/21% licit labeled, temporal data across 49 time steps

---

## **#2 - IBM AML Synthetic Transaction Dataset**
- **Source Platform**: Kaggle
- **Suitability Score**: 92/100
- **Direct URL**: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
- **Dataset Size**: 6 datasets, 8.18GB total
- **Problem Relevance**: High - Designed for traditional banking AML
- **Data Quality**: Excellent - Models complete money laundering cycles
- **License Type**: Open (CDLA-Sharing-1.0)
- **Last Updated**: July 2025
- **Preprocessing Needs**: Minimal - Ready for ML and GNN applications
- **Key Features**: Complete financial ecosystem, placement/layering/integration modeling, multiple institution support, graph neural network ready

---

## **#3 - UCI BitcoinHeist Ransomware Dataset**
- **Source Platform**: UCI ML Repository
- **Suitability Score**: 88/100
- **Direct URL**: https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset
- **Dataset Size**: 2,916,697 rows × 10 features, 225MB
- **Problem Relevance**: High - Ransomware and illicit transaction focus
- **Data Quality**: Good - 10-year Bitcoin transaction coverage (2009-2018)
- **License Type**: Open (CC BY 4.0)
- **Last Updated**: 2019
- **Preprocessing Needs**: Minimal - Clean structured format
- **Key Features**: Ransomware address labeling, transaction pattern features (loops, neighbors, income), 0.3+ Bitcoin threshold filtering

---

## **#4 - Ethereum Fraud Detection Dataset (Vagifa)**
- **Source Platform**: Kaggle
- **Suitability Score**: 82/100
- **Direct URL**: https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset
- **Dataset Size**: ~9,841 observations × 45+ features, 944KB
- **Problem Relevance**: High - Ethereum-specific fraud detection
- **Data Quality**: Good - Comprehensive transaction metadata
- **License Type**: Open (ODbL)
- **Last Updated**: January 2021
- **Preprocessing Needs**: Moderate - Imbalanced dataset requiring balancing
- **Key Features**: Ethereum account-level aggregated features, ERC20 token transactions, contract interactions, fraud binary classification

---

## **#5 - SAML-D Synthetic AML Dataset**
- **Source Platform**: Kaggle
- **Suitability Score**: 79/100
- **Direct URL**: https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml
- **Dataset Size**: 9,504,852 transactions × 12 features, 203MB
- **Problem Relevance**: Medium-High - Traditional AML with crypto applicability
- **Data Quality**: Good - 28 typologies, 15 network structures
- **License Type**: Restricted (CC BY-NC-SA 4.0)
- **Last Updated**: January 2024
- **Preprocessing Needs**: Moderate - 0.1% suspicious transaction imbalance
- **Key Features**: Cross-border transactions, high-risk region indicators, multiple payment types, graphical network flows

---

## **#6 - Bitcoin Sentiment-Augmented Dataset**
- **Source Platform**: Hugging Face
- **Suitability Score**: 75/100
- **Direct URL**: https://huggingface.co/datasets/danilocorsi/LLMs-Sentiment-Augmented-Bitcoin-Dataset
- **Dataset Size**: 25,718 rows, 3.97GB
- **Problem Relevance**: Medium - Price/sentiment correlation for compliance context
- **Data Quality**: Good - 8-year coverage (2016-2024)
- **License Type**: Open (MIT)
- **Last Updated**: 2024
- **Preprocessing Needs**: Moderate - NLP sentiment processing required
- **Key Features**: Bitcoin blockchain metrics, Fear & Greed Index, social sentiment analysis, trading recommendations with confidence scores

---

## **#7 - Google BigQuery Cryptocurrency Datasets**
- **Source Platform**: Google Datasets
- **Suitability Score**: 71/100
- **Direct URL**: https://cloud.google.com/blockchain-analytics/docs/supported-datasets
- **Dataset Size**: Multiple TB-scale datasets, 17 cryptocurrencies
- **Problem Relevance**: Medium - Raw blockchain data requiring processing
- **Data Quality**: Excellent - Daily updated, comprehensive coverage
- **License Type**: Open (Google Cloud terms)
- **Last Updated**: Daily updates
- **Preprocessing Needs**: Extensive - Raw blockchain data requires significant processing
- **Key Features**: Bitcoin, Ethereum, Bitcoin Cash, Dogecoin, Litecoin, Zcash, and 11 others; complete transaction histories, 24-hour update cycle

---

## **#8 - Ethereum Fraud Detection (Rupak Roy)**
- **Source Platform**: Kaggle
- **Suitability Score**: 68/100
- **Direct URL**: https://www.kaggle.com/datasets/rupakroy/ethereum-fraud-detection
- **Dataset Size**: 50+ features, 944KB
- **Problem Relevance**: High - Ethereum fraud focus
- **Data Quality**: Fair - Limited documentation
- **License Type**: Open (CC0 Public Domain)
- **Last Updated**: March 2021
- **Preprocessing Needs**: Moderate - Binary classification ready
- **Key Features**: Ethereum transaction characteristics, ERC20 token metrics, contract interaction details, fraud binary labeling

---

## **#9 - Financial Fraud Dataset (Hugging Face)**
- **Source Platform**: Hugging Face
- **Suitability Score**: 64/100
- **Direct URL**: https://huggingface.co/datasets/amitkedia/Financial-Fraud-Dataset
- **Dataset Size**: Not specified in search results
- **Problem Relevance**: Medium - General financial fraud
- **Data Quality**: Unknown - Limited metadata available
- **License Type**: Unknown
- **Last Updated**: Unknown
- **Preprocessing Needs**: Unknown - Requires investigation
- **Key Features**: General financial fraud detection focus, limited publicly available details

---

## **#10 - Cryptocurrency Scam Dataset**
- **Source Platform**: Kaggle
- **Suitability Score**: 61/100
- **Direct URL**: https://www.kaggle.com/datasets/zongaobian/cryptocurrency-scam-dataset
- **Dataset Size**: Not specified in search results
- **Problem Relevance**: Medium - Phishing/scam detection
- **Data Quality**: Unknown - Limited documentation in search
- **License Type**: Unknown
- **Last Updated**: Recent (based on search results)
- **Preprocessing Needs**: Unknown - URL/text processing likely required
- **Key Features**: Cryptocurrency scam URLs, phishing site collection, fraud website identification

---

## Comparative Analysis Matrix

| Dataset | Bitcoin | Ethereum | Multi-Chain | Graph Analysis | AML Ready | Banking Compliance |
|---------|---------|----------|-------------|----------------|-----------|-------------------|
| Elliptic | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |
| IBM AML | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| BitcoinHeist | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ |
| Ethereum (Vagifa) | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ |
| SAML-D | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |

## Implementation Recommendations

**For Traditional Banking Compliance**: Start with IBM AML Dataset (#2) for comprehensive money laundering pattern modeling, then integrate Elliptic Dataset (#1) for Bitcoin-specific analysis.

**For Multi-Cryptocurrency Coverage**: Combine Google BigQuery datasets (#7) with Elliptic (#1) and Ethereum datasets (#4, #8) for comprehensive cross-chain analysis.

**For Immediate Implementation**: Elliptic Dataset (#1) provides the highest-quality, most research-validated foundation for cryptocurrency AML compliance systems.

**For Advanced Graph Analysis**: Use Elliptic (#1) and IBM AML (#2) datasets together to leverage both real Bitcoin transaction patterns and synthetic money laundering network structures.

The top 3 datasets provide complementary strengths covering Bitcoin graph analysis, traditional banking AML integration, and illicit transaction pattern recognition essential for your cryptocurrency compliance requirements.

---

## Generated by: dataset-exploration-researcher agent
**Date**: 2025-08-25  
**Focus**: Cryptocurrency compliance datasets from UCI, Kaggle, Google Datasets, and Hugging Face