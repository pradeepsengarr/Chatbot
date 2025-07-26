PDF based Question Answering and Summarization System
This project is designed to build a PDF-based Question Answering (QA) and Summarization system using state-of-the-art language models and embeddings. The system allows users to upload multiple PDFs, and it provides precise answers to user questions based on the contents of the documents. Additionally, the system can generate summaries for either entire PDFs or specific topics from the documents.

Key Features:
  Document Ingestion and Chunking: Efficiently loads multiple PDFs, processes their content, and splits them into manageable chunks for faster and more accurate processing.
  Question Answering: Utilizes pre-trained models to generate answers from the uploaded PDFs based on user queries, designed specifically for Chartered Accountants and audit-related questions.
  Summarization: Summarizes entire PDFs or specific subjects from the documents, providing concise and relevant information.
  Embeddings and Retrieval: Uses sentence embeddings for document vectorization and retrieval, ensuring the relevant content is fetched when answering user queries.
  LLM Integration: Leverages models like MBZUAI/LaMini-T5-738M to handle text generation and answer complex questions with detailed references.
  User-friendly Logging: Provides detailed logs of all processing steps, from document ingestion to answer generation and summarization.
Tech Stack:
  LangChain: For document loading, text chunking, and QA chains.
  Transformers: For integrating and fine-tuning pre-trained language models.
  Sentence Transformers: For generating embeddings from text chunks.
  Chroma: For vector store management and efficient document retrieval.
  PyTorch: To support model inference on CPU.
  Python: The entire backend is written in Python, ensuring extensibility and ease of development.
Usage:
  Document Upload: Place PDFs in the designated directory.
  Ask Questions: Query the system with questions related to the uploaded PDFs.
  Generate Summaries: Request summaries for specific topics or entire documents.
