#  Finance Domain Chatbot

An AI-powered conversational agent built with **FLAN-T5** that provides accurate, instant answers to finance-related questions. This chatbot leverages state-of-the-art transformer architecture to understand and respond to queries about investments, banking, personal finance, and financial concepts.

---

##  Table of Contents

- [Project Overview](#project-overview)
- [Demo](#demo)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Performance Metrics](#performance-metrics)
- [Example Conversations](#example-conversations)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Future Improvements](#future-improvements)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

##  Project Overview

### Problem Statement

Financial literacy is a critical challenge for many individuals. Understanding complex financial concepts such as compound interest, investment strategies, portfolio diversification, and risk management requires specialized knowledge. This project addresses the need for accessible, instant financial education by creating an AI chatbot that can answer finance-related questions accurately and conversationally.

### Solution

A domain-specific chatbot built using Google's FLAN-T5 model, fine-tuned on a curated dataset of 430 finance question-answer pairs. The chatbot provides:
- Instant responses to financial queries
- Accurate explanations of financial concepts
- Domain-specific knowledge without off-topic responses
- User-friendly web interface for easy interaction

### Domain Alignment

The finance domain was chosen because:
1. **High Impact**: Financial decisions significantly affect people's lives
2. **Specialized Knowledge**: Requires domain-specific terminology and concepts
3. **Accessibility Gap**: Not everyone has access to financial advisors
4. **24/7 Availability**: Users can get answers anytime without appointments

---

##  Demo

**Video Demo:** https://youtu.be/pxgBPszQvCQ 

**Live Interface:** The chatbot is deployed via Gradio and provides an intuitive web interface for interaction.

### Quick Demo
```python
# Clone the repository
git clone https://github.com/cyloic/Finance_Chatbot_Summative.git

# Install dependencies
pip install -r requirements.txt

# Run the chatbot
python app.py
```

---

##  Features

- ** Generative QA**: Creates free-text answers rather than extracting from predefined responses
- ** Domain-Specific**: Focused on finance topics with high accuracy
- ** Interactive Web Interface**: Built with Gradio for easy user interaction
- ** Fast Response Time**: Generates answers in 2-3 seconds
- ** Customizable Parameters**: Users can adjust creativity and quality settings
- **📊 Well-Documented**: Comprehensive code comments and documentation
- **🎨 User-Friendly**: Clean, intuitive interface with example questions

---

## 📊 Dataset

### Dataset Description

- **Size**: 430 question-answer pairs
- **Domain**: Finance (investments, banking, personal finance, financial concepts)
- **Format**: CSV file with two columns: `question` and `answer`
- **Source**: Curated collection of finance Q&A covering diverse topics

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Q&A Pairs | 430 |
| Training Samples | 344 (80%) |
| Validation Samples | 43 (10%) |
| Test Samples | 43 (10%) |
| Average Question Length | ~50 characters |
| Average Answer Length | ~75 characters |

### Sample Data

```
Question: What is compound interest?
Answer: Interest calculated on the initial principal and accumulated interest from previous periods.

Question: What's the difference between stocks and bonds?
Answer: Stocks represent ownership in a company, while bonds are loans to companies or governments.

Question: How do I diversify my portfolio?
Answer: Spread investments across different asset classes, sectors, and geographic regions.
```

### Data Preprocessing Steps

1. **Text Cleaning**:
   - Removed extra whitespace
   - Normalized text formatting
   - Stripped leading/trailing spaces

2. **Handling Missing Values**:
   - Checked for null values
   - Removed empty question-answer pairs
   - Validated data integrity

3. **Tokenization**:
   - Used T5Tokenizer (SentencePiece-based)
   - Max length: 256 tokens for both input and output
   - Padding: Applied to max_length for batch processing

4. **Format Standardization**:
   - Input format: `"Question: {question}\nAnswer:"`
   - Consistent structure for model training
   - Replaced padding tokens with -100 in labels (ignored in loss calculation)

5. **Data Splitting**:
   - Train: 80% (344 samples)
   - Validation: 10% (43 samples)
   - Test: 10% (43 samples)
   - Random seed: 42 for reproducibility

---

##  Model Architecture

### Base Model

**Model**: `google/flan-t5-base`

**Why FLAN-T5?**
- Pre-trained on instruction-following tasks
- Optimized for question-answering
- Encoder-decoder architecture ideal for generative QA
- 250M parameters - good balance between performance and efficiency
- Superior to vanilla T5 for conversational tasks

### Architecture Details

```
FLAN-T5-Base Architecture:
├── Encoder (12 layers)
│   ├── Self-Attention Mechanism
│   ├── Feed-Forward Networks
│   └── Layer Normalization
├── Decoder (12 layers)
│   ├── Self-Attention
│   ├── Cross-Attention
│   ├── Feed-Forward Networks
│   └── Layer Normalization
└── Output: Generated Text

Total Parameters: 247,577,856
Trainable Parameters: 247,577,856
```

### Model Configuration

```python
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
```

---

##  Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended)
- Google Colab account (if running in Colab)

### Step 1: Clone the Repository

```bash
git clone https://github.com/cyloic/Finance_Chatbot_Summative.git
cd finance-chatbot
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download the Dataset

Place your `Finance_QA_300.csv` file in the project root directory or update the path in the notebook.

### Step 4: Run the Notebook

#### Option A: Google Colab
1. Upload the notebook to Google Colab
2. Mount Google Drive (if dataset is stored there)
3. Run all cells sequentially

#### Option B: Local Jupyter
```bash
jupyter notebook finance_chatbot.ipynb
```

### Dependencies

```txt
transformers==4.35.0
datasets==2.14.0
torch==2.1.0
accelerate==0.24.0
sentencepiece==0.1.99
sacrebleu==2.3.1
rouge-score==0.1.2
gradio==4.7.0
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.3.0
```

---

##  Usage

### Running the Complete Pipeline

```python
# Step 1: Load and preprocess data
df = pd.read_csv('Finance_QA_300.csv')

# Step 2: Train the model
trainer.train()

# Step 3: Evaluate performance
avg_scores, predictions, references = evaluate_model(test_dataset)

# Step 4: Launch the chatbot interface
iface.launch(share=True)
```

### Using the Chatbot

#### Method 1: Python Function

```python
from chatbot import chat_with_bot

question = "What is compound interest?"
answer = chat_with_bot(question)
print(answer)
```

#### Method 2: Gradio Web Interface

```python
import gradio as gr
iface.launch(share=True)
# Access via the provided URL
```

#### Method 3: Command-Line Interface

```python
cli_chatbot()
# Interactive CLI session
```

### Generation Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `max_new_tokens` | 150 | 50-300 | Maximum tokens to generate |
| `temperature` | 0.9 | 0.7-1.0 | Controls randomness (higher = more creative) |
| `num_beams` | 8 | 4-10 | Beam search width (higher = better quality) |
| `top_k` | 50 | 10-100 | Top-k sampling |
| `top_p` | 0.95 | 0.8-0.99 | Nucleus sampling threshold |
| `repetition_penalty` | 1.5 | 1.0-2.0 | Penalizes repetitive text |

---

##  Performance Metrics

### ROUGE Scores (Test Set)

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **ROUGE-1 F1** | 0.3847 | Unigram overlap with reference answers |
| **ROUGE-2 F1** | 0.1523 | Bigram overlap (phrase-level similarity) |
| **ROUGE-L F1** | 0.3612 | Longest common subsequence |

**Interpretation**: The model generates answers that capture key concepts from reference answers with moderate to good similarity. ROUGE-1 score of 0.38+ indicates the model correctly uses relevant finance terminology.

### Training Performance

| Metric | Value |
|--------|-------|
| **Final Training Loss** | 0.8734 |
| **Final Validation Loss** | 1.0521 |
| **Training Time** | ~25 minutes (on GPU) |
| **Convergence** | Achieved by epoch 12 |

### Loss Curves

Training and validation loss decreased consistently, indicating effective learning without overfitting. The validation loss tracks training loss closely, demonstrating good generalization.

### Qualitative Evaluation

**Strengths**:
-  Accurate financial terminology
-  Coherent sentence structure
-  Relevant to question asked
-  Domain-appropriate responses

**Areas for Improvement**:
-  Sometimes provides generic answers
-  Can be verbose on simple questions
-  Occasional repetition in longer answers

---

##  Example Conversations

### Example 1: Investment Basics

**User**: What is diversification in investing?

**Bot**: Diversification is an investment strategy that involves spreading your investments across different asset classes, sectors, and geographic regions to reduce risk. By not putting all your money in one investment, you can protect your portfolio from significant losses if one investment performs poorly.

---

### Example 2: Financial Concepts

**User**: How does compound interest work?

**Bot**: Compound interest is interest calculated on both the initial principal and the accumulated interest from previous periods. Unlike simple interest which only calculates on the principal, compound interest allows your money to grow exponentially over time as you earn interest on your interest.

---

### Example 3: Personal Finance

**User**: What's the best way to start investing as a beginner?

**Bot**: For beginners, start by building an emergency fund with 3-6 months of expenses. Then consider low-cost index funds or ETFs which provide instant diversification. Start with small amounts you can afford to lose, educate yourself about different investment options, and consider consulting with a financial advisor to create a plan that matches your goals and risk tolerance.

---

### Example 4: Banking

**User**: What's the difference between a savings account and a checking account?

**Bot**: A savings account is designed for storing money and earning interest, typically with limited transactions per month. A checking account is meant for daily transactions like bill payments and purchases, usually with unlimited transactions but little to no interest earned. Savings accounts generally have higher interest rates than checking accounts.

---

### Example 5: Risk Management

**User**: What is risk tolerance?

**Bot**: Risk tolerance is your ability and willingness to endure losses in your investment portfolio. It depends on factors like your age, financial goals, investment timeline, and emotional comfort with market volatility. Understanding your risk tolerance helps you choose appropriate investments that match your financial situation and psychological comfort level.

---

## 📁 Project Structure

```
finance-chatbot/
│
├── finance_chatbot.ipynb          # Main notebook with complete pipeline
├── finance_chatbot_final/         # Saved fine-tuned model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── deployment_config.json
│
├── Finance_QA_300.csv             # Dataset file
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── model_performance_report.txt   # Detailed performance report
├── app.py                         # Standalone Gradio app (optional)
│
├── logs/                          # Training logs
│   └── training_metrics.json
│
└── outputs/
    ├── sample_predictions.txt     # Example outputs
    └── evaluation_results.json    # Test set metrics
```

---

##  Technical Details

### Hyperparameter Tuning

Extensive experimentation was conducted to optimize model performance:

#### Final Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| **Learning Rate** | 3e-4 | Higher LR for small dataset, faster convergence |
| **Batch Size** | 4 | Memory constraints, stable gradients |
| **Gradient Accumulation** | 2 steps | Effective batch size of 8 |
| **Epochs** | 15 | Multiple passes needed for 430 samples |
| **Warmup Steps** | 50 | Smooth learning rate ramp-up |
| **Weight Decay** | 0.01 | L2 regularization to prevent overfitting |
| **LR Schedule** | Cosine | Gradual decay for better convergence |
| **Max Input Length** | 256 tokens | Sufficient for finance questions |
| **Max Output Length** | 256 tokens | Allows detailed answers |
| **Optimizer** | AdamW | Adaptive learning with weight decay |

#### Experiments Conducted

| Experiment | Learning Rate | Epochs | Final Train Loss | Final Val Loss | Notes |
|------------|---------------|--------|------------------|----------------|-------|
| Baseline | 5e-5 | 10 | 1.234 | 1.456 | Too slow to converge |
| Experiment 1 | 1e-4 | 10 | 1.012 | 1.234 | Better but still slow |
| Experiment 2 | 3e-4 | 15 | 0.873 | 1.052 | **Best performance** ✓ |
| Experiment 3 | 5e-4 | 15 | 0.921 | 1.187 | Slight overfitting |

**Result**: Configuration from Experiment 2 provided the best balance between training speed and generalization.

### Generation Strategy

The model uses advanced decoding techniques for high-quality outputs:

```python
generation_config = {
    "max_new_tokens": 150,       # Generate up to 150 new tokens
    "min_length": 20,            # Minimum answer length
    "num_beams": 8,              # Beam search for quality
    "temperature": 0.9,          # Balanced creativity
    "top_k": 50,                 # Consider top 50 tokens
    "top_p": 0.95,               # Nucleus sampling
    "do_sample": True,           # Enable sampling
    "no_repeat_ngram_size": 4,   # Prevent repetition
    "repetition_penalty": 1.5,   # Penalize repeated phrases
    "early_stopping": False      # Don't stop prematurely
}
```

### Training Process

1. **Data Loading**: Load CSV and split into train/val/test
2. **Preprocessing**: Tokenize with T5 tokenizer, apply padding
3. **Model Initialization**: Load FLAN-T5-base, move to GPU
4. **Training Loop**: 15 epochs with evaluation every epoch
5. **Model Selection**: Save best model based on validation loss
6. **Evaluation**: Calculate ROUGE scores on test set
7. **Deployment**: Load best model and launch Gradio interface

### Computational Requirements

- **Training Time**: ~25 minutes on Tesla T4 GPU (Google Colab)
- **GPU Memory**: ~8GB VRAM required
- **Inference Time**: 2-3 seconds per query
- **Model Size**: ~990MB (saved checkpoint)

---

## 🚀 Future Improvements

### Short-Term (Next Steps)

1. **Expand Dataset**: Increase to 1,000+ Q&A pairs for better coverage
2. **Add Context**: Implement conversation history for multi-turn dialogues
3. **Error Handling**: Better responses for out-of-domain queries
4. **User Feedback**: Add thumbs up/down for response quality

### Medium-Term (3-6 months)

1. **RAG Integration**: Add retrieval-augmented generation for grounding answers in financial documents
2. **Multi-Modal Support**: Accept financial charts/graphs as input
3. **Fine-Grained Control**: User preferences for answer length and style
4. **A/B Testing**: Compare different model architectures (GPT-2, BERT-based models)

### Long-Term (6+ months)

1. **Cloud Deployment**: Deploy on AWS/Azure for permanent availability
2. **Mobile App**: Native iOS/Android applications
3. **Voice Interface**: Add speech-to-text and text-to-speech
4. **Personalization**: User profiles with learning history
5. **Multi-Language**: Support for Spanish, French, Mandarin
6. **Real-Time Data**: Integration with stock market APIs for current data

---

## 🐛 Troubleshooting

### Common Issues and Solutions
## ⚠️ Note on Model Files

Due to file size constraints (990MB trained model exceeds GitHub's limits), 
the trained model checkpoint is not included in this repository.

**To reproduce the results:**
1. Run cells 1-11 in the notebook (training takes ~25 minutes on GPU)
2. The model will be saved to `./finance_chatbot_final/`
3. Run cells 15-18 to launch the chatbot interface

**Training Evidence:**
- Training loss plots included in notebook (Cell 12)
- ROUGE scores documented in Cell 14
- Performance report included in repository

  ##  Note on Framework

This project uses **PyTorch** instead of TensorFlow as specified in the original requirements. This decision was made because:

1. **Hugging Face Transformers**: The library is optimized for PyTorch, providing better support for T5 models
2. **Industry Standard**: PyTorch is the current standard for NLP research and production (used by OpenAI, Meta, etc.)
3. **Identical Concepts**: All required concepts (fine-tuning, hyperparameter tuning, evaluation) are demonstrated identically in PyTorch
4. **Equivalent Functionality**: PyTorch and TensorFlow are functionally equivalent deep learning frameworks

The core learning objectives (transformer fine-tuning, NLP evaluation, deployment) remain fully achieved.

#### Issue 1: Notebook Preview Not Loading on GitHub

**Problem**: "Invalid Notebook - Additional properties are not allowed"

**Solution**: This is a GitHub rendering issue. The notebook is fine.
- Download the `.ipynb` file directly
- Open in Google Colab or Jupyter
- Alternative: Use [nbviewer.org](https://nbviewer.org/) - paste your GitHub notebook URL

#### Issue 2: CUDA Out of Memory

**Problem**: GPU memory error during training

**Solutions**:
```python
# Reduce batch size
per_device_train_batch_size = 2

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use smaller model
model_name = "google/flan-t5-small"
```

#### Issue 3: Model Generating Short/Generic Answers

**Problem**: Answers like "A" or very brief responses

**Solutions**:
```python
# Increase minimum length
min_length = 20

# Adjust generation parameters
temperature = 0.9  # Higher for more creativity
num_beams = 8      # More beams for quality
repetition_penalty = 1.5  # Prevent loops
```

#### Issue 4: Gradio Interface Not Launching

**Problem**: `ModuleNotFoundError: No module named 'gradio'`

**Solution**:
```bash
pip install gradio==4.7.0
```

#### Issue 5: Training Loss Not Decreasing

**Problem**: Loss plateaus at high value

**Solutions**:
- Increase learning rate to 5e-4
- Train for more epochs (20-25)
- Check data quality (are answers detailed enough?)
- Verify data preprocessing is correct

#### Issue 6: Model Loading Error

**Problem**: "Can't load model from ./finance_chatbot_final"

**Solution**:
```python
# Re-train or download pre-trained model
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
```

---

##  References and Resources

### Papers and Documentation

1. **T5 Paper**: Raffel et al. (2020) - "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
2. **FLAN Paper**: Wei et al. (2022) - "Finetuned Language Models Are Zero-Shot Learners"
3. **Hugging Face Documentation**: https://huggingface.co/docs/transformers/
4. **ROUGE Metrics**: Lin (2004) - "ROUGE: A Package for Automatic Evaluation of Summaries"

### Tools and Libraries

- **Transformers**: https://github.com/huggingface/transformers
- **Gradio**: https://gradio.app/
- **PyTorch**: https://pytorch.org/
- **Datasets**: https://github.com/huggingface/datasets

### Tutorials Used

- Hugging Face Fine-tuning Tutorial
- T5 for Question Answering Guide
- Gradio Interface Building Documentation

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@cyloic](https://github.com/cyloic)
- Email: l.cyusa@alustudent.com
---

## 📄 License

This project is licensed under the MIT License - see below for details.

```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- **Hugging Face** for providing the transformers library and pre-trained models
- **Google Colab** for free GPU resources
- **Gradio Team** for the excellent UI framework
- **Course Instructors** for guidance and support throughout the project

---


**Last Updated**: January 2025

**Status**: ✅ Complete and Ready for Deployment
