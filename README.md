# AgenticAI

## Bootcamp Learning

### NLP

- **Tokenization**
  - Corus
  - Sentence
  - Word
  - Vocab
  - NLTK Tokenization
- **NLTK Stem - Step Word**
  - Porter Stemmer
  - Regex Stemmer
  - Snowball Stemmer
  - Wordnet Lemmatizer - Root Word
- **POS (Part of Speech) Tagging**
  - CC, CD, EX, V, N
  - Stop words
  - NLTK POS_TAG
- **Named Entity Recognition**
- **Word to Vector**
  - One Hot Encoding
  - Bag of Words
  - TF-IDF (Term Frequency - Inverse Document Frequency)
    - **Advantages**
      - Intuitive
      - Fixed size - vocab size
      - Word importance
    - **Disadvantages**
      - OV - Out of Vocabulary
      - Sparsity still exists
- **Word Embeddings**
  - Representation of words for text analysis
  - Feature representation of words
  - **Types**
    - Count of Frequency
    - One Hot Encoding
    - Bag of Words
    - TF-IDF
    - Deep Learning Trained Model - Word2Vec
      - CBOW - Continuous Bag of Words - ANN
      - Fully Connected Neural Network
      - Cosine Similarity - Angle between two vectors
      - Skipgram
      - Avg Word to Vec
      - Gensim Library

### ML
- Stats tool -> Analyze, visualize, predict the data

### DL
- Multi-layer neural network
- **Attention is All You Need**
- Transformer

### Generative AI
- AI Agents
- Agentic AI

## Environment Setup
```sh
conda create -p venv python==3.12
conda activate venv/
pip install -r requirements.txt
pip install ipykernel
```

## What is LangChain?
Common Gen AI framework that integrates any LLM model.

### Integrations
- **LangSmith**
  - Debug
  - Monitor
- **LangGraph**
- **LangServe**

### Ollama → LLMs Local → Open Source Local
```sh
ollama run llama3.2:1b
```

## Data Ingestion Techniques
- **Loaders**
  - TextLoader
  - PyPdfLoader
  - WebBaseLoader
  - ArxivLoader
  - WikipediaLoader

## Data Transformer Techniques
- Recursive Character Splitter
- Character Text Splitter
- HTML Text Splitter
- RecursiveJsonSplitter

## Embedding Techniques - Convert Text to Vector
- OllamaEmbeddings
- OpenAIEmbeddings
- Hugging Face Embeddings

## Vector Stores
- Faiss
- Chroma

## Groq

