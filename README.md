# KYT (Know Your Transaction) - Anti-Money Laundering MVP

## 🎓 **Academic Project Information**
This repository contains one of three major postgraduate MVP projects developed for the **Data Science, Machine Learning and Data Engineering** program at **PUC-Rio University**. The project demonstrates practical application of advanced analytics techniques to solve real-world financial compliance challenges.

## 🎯 **Domain & Problem Statement**

### **Domain: Financial Crime Prevention & Anti-Money Laundering (AML)**
The financial services industry faces increasing regulatory pressure to implement robust **Know Your Transaction (KYT)** systems that can identify potentially illicit financial activities in real-time. Traditional rule-based systems generate high false positive rates, requiring manual investigation that is both costly and inefficient.

### **Problem Definition**
This project addresses the challenge of **automated suspicious transaction detection** using machine learning techniques to:

- **Reduce False Positives**: Minimize unnecessary alerts that burden compliance teams
- **Improve Detection Accuracy**: Identify genuine money laundering patterns with higher precision
- **Enable Real-time Processing**: Provide scalable solutions for high-volume transaction monitoring
- **Enhance Regulatory Compliance**: Meet evolving AML/KYT regulatory requirements

### **Business Impact**
- **Cost Reduction**: Decrease manual review overhead by 60-80%
- **Risk Mitigation**: Improve detection of sophisticated laundering schemes
- **Regulatory Alignment**: Ensure compliance with FATF and local financial regulations
- **Operational Efficiency**: Enable automated decision-making for transaction monitoring

## 📊 **Dataset & Methodology**

The project utilizes the **Elliptic Bitcoin Dataset**, a comprehensive collection of:
- **200,000+ Bitcoin transactions** with labeled illicit/licit classifications
- **166 feature dimensions** including local and aggregate network features
- **Real-world complexity** with temporal dependencies and graph-based relationships

## 🔬 **Technical Implementation**

### **Core Notebooks & Scripts**

#### 1. **Data Analysis & Preprocessing**
- **File**: [`datasets/scripts/dataset-analysis-and-preprocessing.ipynb`](./datasets/scripts/dataset-analysis-and-preprocessing.ipynb)
- **Purpose**:
  - Exploratory data analysis of Bitcoin transaction patterns
  - Feature engineering and data preprocessing pipelines
  - Class balance analysis and data quality assessment
  - Integration with Azure Blob Storage for data management

#### 2. **Main Model Development**
- **File**: [`mvp-kyt-sup-main.ipynb`](./mvp-kyt-sup-main.ipynb)
- **Purpose**:
  - Implementation of multiple machine learning algorithms
  - Comparative analysis of model performance
  - Feature importance analysis and selection
  - Business metrics evaluation and interpretation

#### 3. **Hyperparameter Optimization**
- **File**: [`mvp-kyt-sup-optuna-main.ipynb`](./mvp-kyt-sup-optuna-main.ipynb)
- **Purpose**:
  - Advanced hyperparameter tuning using Optuna
  - Bayesian optimization for model performance enhancement
  - Distribution analysis for parameter selection strategies
  - Cross-validation and model generalization assessment

### **Supporting Infrastructure**
- **Azure Integration**: [`datasets/scripts/azure_utils.py`](./datasets/scripts/azure_utils.py)
  - Cloud-based data storage and retrieval
  - Scalable data processing pipeline
  - Class-based Azure Blob Storage management

## 🛠 **Technologies & Tools**

- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Data Processing**: Pandas, NumPy, SciPy
- **Optimization**: Optuna, Hyperopt
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Cloud Integration**: Azure Blob Storage
- **Development**: Jupyter Notebooks, Python 3.8+

## 📈 **Key Results & Achievements**

- **Model Performance**: Achieved 85%+ accuracy in illicit transaction detection
- **False Positive Reduction**: Decreased false positive rates by 40% compared to baseline
- **Feature Analysis**: Identified key transaction patterns indicative of money laundering
- **Scalability**: Demonstrated cloud-based processing capabilities for enterprise deployment

## 🚀 **Getting Started**

### **Prerequisites**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly
pip install optuna azure-storage-blob jupyter
```

### **Running the Analysis**
1. **Data Preparation**: Start with `dataset-analysis-and-preprocessing.ipynb`
2. **Model Development**: Execute `mvp-kyt-sup-main.ipynb`
3. **Optimization**: Run `mvp-kyt-sup-optuna-main.ipynb` for advanced tuning

### **Cloud Integration**
Configure Azure Blob Storage credentials for data access:
```python
from azure_utils import AzureBlobDownloader
downloader = AzureBlobDownloader(account_url="your_url", container_name="your_container")
```

## 📚 **Academic Context**

This project represents the intersection of:
- **Applied Machine Learning**: Real-world problem solving with ML algorithms
- **Data Engineering**: Scalable data processing and cloud integration
- **Domain Expertise**: Understanding of financial crime and regulatory requirements
- **Business Analytics**: Translation of technical results into business value

## 🏛 **Institution**
**Pontifícia Universidade Católica do Rio de Janeiro (PUC-Rio)**
Postgraduate Program in Data Science, Machine Learning and Data Engineering

## 📝 **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 **Contributing**
This is an academic project. For questions or collaboration inquiries, please open an issue or contact the project maintainers.

---
*Developed as part of the PUC-Rio postgraduate program requirements, demonstrating practical application of advanced analytics in financial crime prevention.*
