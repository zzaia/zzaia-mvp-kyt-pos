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

### **Business Context**
- **Cost Reduction**: Potential to decrease manual review overhead through automated detection
- **Risk Mitigation**: Enhanced detection of sophisticated laundering schemes
- **Regulatory Alignment**: Support compliance with FATF and local financial regulations
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
  - Advanced hyperparameter tuning using Optuna
  - Bayesian optimization for model performance enhancement
  - Implementation of multiple machine learning algorithms
  - Comparative analysis of model performance
  - Feature importance analysis and selection
  - Business metrics evaluation and interpretation
  - Cross-validation and model generalization assessment
  - Distribution analysis for parameter selection strategies

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

- **Model Performance**: Achieved 99.90% PR-AUC score with TabNet as the best-performing model
- **Algorithm Comparison**: Comprehensive evaluation of 20 machine learning algorithms 
- **Feature Engineering**: Successfully reduced dimensionality from 166 to 59 features using PCA
- **Real-world Application**: Processed 157,205 unlabeled transactions, identifying 10,277 potentially illicit cases
- **Model Ranking**: TabNet (97.50%) > FNN (96.44%) > Vote-Soft (96.28%) deep learning excels in financial pattern recognition followed by traditional SVC and FNN combined
- **Scalability**: Demonstrated cloud-based processing capabilities with Azure Blob Storage integration
- **Academic Contribution**: Validated machine learning approaches for cryptocurrency transaction risk assessment

## 🤖 AI-Assisted Development

This project showcases the integration of AI-powered development workflows using a **customized agentic system** built on Claude Code.

### Custom AI Workspace
- **Repository**: [zzaia-agentic-workspace](https://github.com/zzaia/zzaia-agentic-workspace)
- **Toolset**: Claude Code with custom agents, slash commands, and templates 
- **Workflow**: Multi-repository management, git worktrees, automated task orchestration

## 🚀 **Getting Started**

### **Quick Start with Google Colab**

Run the main notebook directly in your browser without any local setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zzaia/zzaia-mvp-kyt-pos/blob/main/mvp-kyt-sup-main.ipynb)

**Benefits of using Colab:**
- ✅ No local installation required
- ✅ Free GPU/TPU access for faster training
- ✅ Pre-configured Python environment
- ✅ Direct integration with GitHub
- ✅ Easy sharing and collaboration

**Note**: The notebook will automatically clone the repository and install required dependencies when run in Colab.

### **Prerequisites (Local Setup)**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly kaggle
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

##
> *"I can do all this through him who gives me strength."*
>
> **— Philippians 4:13**
