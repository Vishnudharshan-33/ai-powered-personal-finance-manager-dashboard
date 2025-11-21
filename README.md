# ai-powered-personal-finance-manager-dashboard
AI-powered personal finance manager with an intelligent chatbot and real-time dashboard – track expenses, categorize transactions, get smart insights, and chat with your money in seconds.


# 💰 Personal Finance Tracker with AI

An intelligent finance management system that automatically categorizes bank transactions using AI, visualizes spending patterns, and provides a conversational chatbot to query your financial data.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![AI](https://img.shields.io/badge/AI-LangChain-orange.svg)

## 📋 Overview

This project transforms raw bank transaction data into actionable insights through three core components:

1. **AI-Powered Categorization** - Automatically classifies transactions into categories (Dining, Transport, Entertainment, etc.)
2. **Interactive Dashboard** - Visual analytics with pie charts, bar graphs, and trend analysis
3. **Finance Chatbot** - Ask natural language questions about your spending habits

## ✨ Features

- 🤖 **Automated Categorization**: Uses LLMs (Ollama/OpenAI) to intelligently categorize transactions
- 📊 **Visual Analytics**: Interactive Plotly charts showing spending breakdown by category and month
- 💬 **Natural Language Queries**: Ask questions like "What's my biggest expense?" or "How much did I spend in March?"
- 🔍 **Semantic Search**: FAISS-powered vector search for instant data retrieval
- 📈 **Trend Analysis**: Track spending patterns over time
- 🎯 **Batch Processing**: Efficient handling of large transaction datasets

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **AI/LLM** | LangChain, Ollama/OpenAI |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Panel |
| **Search** | FAISS (Facebook AI Similarity Search) |
| **Web Framework** | Flask |
| **Embeddings** | OpenAI Embeddings |
| **Development** | Jupyter Notebook |

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- Ollama installed (for local AI) OR OpenAI API key

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/finance-tracker.git
cd finance-tracker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install and setup Ollama
ollama pull llama2
```

## 🚀 Quick Start

### 1. Prepare Your Data
Export bank transactions as CSV with columns:
- `Date` - Transaction date
- `Name / Description` - Merchant name
- `Amount` - Transaction amount
- `Expense/Income` - Transaction type

Save as `transactions_2023_2024.csv`

### 2. Categorize Transactions
```bash
jupyter notebook categorize_expenses.ipynb
```
Run all cells to auto-categorize transactions using AI.

**Output**: `combined_transactions_categories.csv`

### 3. View Dashboard
```bash
jupyter notebook dashboard.ipynb
```
Run cells to generate interactive visualizations.

### 4. Launch Chatbot
```bash
python chatbot.py
```
Open browser to `http://localhost:5000` and start asking questions!

## 📂 Project Structure

```
finance-tracker/
├── categorize_expenses.ipynb    # AI categorization workflow
├── dashboard.ipynb               # Data visualization dashboard
├── chatbot.py                    # Flask-based chatbot server
├── templates/
│   └── bot_1.html               # Chatbot UI
├── transactions_2023_2024.csv   # Input: Raw bank data
├── combined_transactions_categories.csv  # Output: Categorized data
├── complete_financial_analysis.txt  # Text summary for chatbot
└── requirements.txt              # Python dependencies
```

## 💡 How It Works

### Categorization Pipeline
1. **Extract Unique Merchants**: From 1000s of transactions, identify unique merchant names
2. **Batch Processing**: Group merchants into batches of 30
3. **AI Classification**: LLM categorizes each batch based on merchant name patterns
4. **Merge Results**: Apply categories back to all matching transactions

### Dashboard Generation
1. **Data Aggregation**: Group transactions by category/month
2. **Visualization**: Create interactive Plotly charts
3. **Layout**: Arrange with Panel for responsive design

### Chatbot Architecture
1. **Text Chunking**: Split financial analysis into searchable segments
2. **Embeddings**: Convert text to vector representations
3. **FAISS Indexing**: Create fast similarity search index
4. **Query Flow**: Question → Embedding → Search → Context → LLM → Answer

## 🎯 Use Cases

- **Personal Finance**: Track personal spending habits and identify savings opportunities
- **Budget Planning**: Analyze historical data to set realistic budgets
- **Tax Preparation**: Categorized expenses simplify tax filing
- **Financial Goals**: Monitor progress toward savings/investment targets
- **Expense Reports**: Generate reports for reimbursements or audits

## 📊 Sample Questions for Chatbot

```
"What's my total spending this year?"
"Which category do I spend most on?"
"How much did I spend on dining in June?"
"Am I spending more or less than last month?"
"What's my average monthly expense?"
"Show me my largest transactions"
```

## ⚙️ Configuration

### Using OpenAI (Fast, Paid)
```python
# In categorize_expenses.ipynb
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key="your-key")
```

### Using Ollama (Free, Local)
```python
# In categorize_expenses.ipynb
from langchain_community.llms import Ollama
llm = Ollama(model="llama2")
```

## 🔧 Customization

### Add Custom Categories
Edit the prompt in `categorize_batch()`:
```python
prompt = f"""Categorize using: Groceries, Dining, Transport, 
Entertainment, Shopping, Bills, Healthcare, Education, Fitness, Travel"""
```

### Adjust Visualization
Modify chart parameters in `dashboard.ipynb`:
```python
# Change color scheme
color_scale = px.colors.sequential.Blues

# Adjust chart size
fig.update_layout(width=800, height=600)
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| CSV encoding error | Save CSV as UTF-8 encoding |
| Ollama connection failed | Ensure Ollama is running: `ollama serve` |
| Charts not displaying | Install Jupyter extensions: `jupyter nbextension enable --py widgetsnbextension` |
| Categories are NaN | Check merchant name matching, re-run categorization |

## 📈 Performance

- **Categorization**: ~124 unique merchants in <30 seconds (OpenAI) / ~5 minutes (Ollama)
- **Dashboard**: Renders 1000+ transactions instantly
- **Chatbot**: <2 second response time with FAISS search

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📝 License

MIT License - feel free to use for personal or commercial projects.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - LLM framework
- [Plotly](https://plotly.com/) - Interactive visualizations
- [FAISS](https://github.com/facebookresearch/faiss) - Similarity search
- [Ollama](https://ollama.ai/) - Local LLM inference

## 📧 Contact

For questions or suggestions, open an issue or reach out at [your-email@example.com]

---

⭐ If you find this project useful, please star the repository!
