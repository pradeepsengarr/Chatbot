import os
import base64
import torch
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Set up logging
logging.basicConfig(level=logging.INFO)

device = torch.device('cpu')  # Use CPU for model inference
persist_directory = "db"  # Directory for Chroma database
uploaded_files_dir = "uploaded_files"  # Directory containing uploaded PDF files

# Load the pre-trained model and tokenizer
checkpoint = "MBZUAI/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Debugging configurations
import math

def data_ingestion():
    """Function to load PDFs and create embeddings with improved error handling and efficiency."""
    try:
        logging.info("Starting data ingestion")

        # Create the folder if it doesn't exist
        if not os.path.exists(uploaded_files_dir):
            os.makedirs(uploaded_files_dir)

        documents = []  # List to store the extracted text from the documents
        for filename in os.listdir(uploaded_files_dir):
            if filename.endswith(".pdf"):
                file_path = os.path.join(uploaded_files_dir, filename)
                logging.info(f"Processing file: {file_path}")
                
                loader = PDFMinerLoader(file_path)
                
                # Load the document and extract text content
                loaded_docs = loader.load()
                
                # Ensure that documents are not empty
                for doc in loaded_docs:
                    if hasattr(doc, 'page_content') and len(doc.page_content.strip()) > 0:
                        documents.append(doc)
                    else:
                        logging.warning(f"Skipping empty or non-text PDF: {file_path}")

        if not documents:
            logging.error("No valid documents found to process.")
            return

        logging.info(f"Total valid documents: {len(documents)}")
        
        # Split the loaded documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        logging.info(f"Total text chunks created: {len(texts)}")
        
        # Ensure there are text chunks to process
        if not texts:
            logging.error("No valid text chunks to create embeddings.")
            return

        # Create embeddings for the text chunks
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Split the texts into smaller batches
        MAX_BATCH_SIZE = 5461  # Chroma max batch size
        total_batches = math.ceil(len(texts) / MAX_BATCH_SIZE)
        
        logging.info(f"Processing {len(texts)} text chunks in {total_batches} batches...")

        # Create and persist a Chroma vector store
        db = None
        for i in range(total_batches):
            batch_start = i * MAX_BATCH_SIZE
            batch_end = min((i + 1) * MAX_BATCH_SIZE, len(texts))
            text_batch = texts[batch_start:batch_end]
            
            logging.info(f"Processing batch {i + 1}/{total_batches}, size: {len(text_batch)}")

            if db is None:
                # Initialize the Chroma store on the first batch
                db = Chroma.from_documents(text_batch, embeddings, persist_directory=persist_directory)
            else:
                # Add the next batch to the existing vector store
                db.add_documents(text_batch)

        db.persist()
        logging.info("Data ingestion completed successfully")
        
    except Exception as e:
        logging.error(f"Error during data ingestion: {str(e)}")
        raise


def llm_pipeline():
    """Function to set up the language model pipeline."""
    logging.info("Setting up LLM pipeline")
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
        device=device
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("LLM pipeline setup complete")
    return local_llm

def qa_llm():
    """Function to set up the question-answering chain."""
    logging.info("Setting up QA model")
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = db.as_retriever()  # Set up the retriever for the vector store
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    logging.info("QA model setup complete")
    return qa
def generate_summary(summary_type, section=None):
    """Generate a summary for the whole document or a specific section."""
    if summary_type == "full":
        # Logic for summarizing the entire document
        logging.info("Generating summary for the entire document...")
        return "Here is the summary of the entire document..."

    elif summary_type == "section" and section:
        # Logic for summarizing a specific section/topic
        logging.info(f"Generating summary for section: {section}")
        return f"Here is the summary for the section on {section}..."

    return "Unable to generate the summary."


def process_answer(user_question):
    """Process user input to generate an answer or a summary if requested."""
    try:
        logging.info("Processing user question")
        qa = qa_llm()  # Get the QA chain

        # Detect if the user is asking for a summary
        if "summary" in user_question.lower():
            if "whole document" in user_question.lower() or "entire pdf" in user_question.lower():
                return generate_summary("full")
            else:
                return generate_summary("section", user_question)
        
        # Otherwise, handle as a normal question
        tailored_prompt = f"""
        You are an expert chatbot designed to assist Chartered Accountants (CAs) in the field of audits. 
        Your goal is to provide accurate and comprehensive answers to any questions related to audit policies, procedures, 
        and accounting standards based on the provided PDF documents. 
        Please respond effectively and refer to the relevant standards and policies whenever applicable.

        User question: {user_question}
        """

        generated_text = qa({"query": tailored_prompt})
        answer = generated_text['result']

        if "not provide" in answer or "no information" in answer:
            return "The document does not provide sufficient information to answer your question."

        logging.info("Answer generated successfully")
        return answer

    except Exception as e:
        logging.error(f"Error during answer generation: {str(e)}")
        raise

def process_summary(user_input):
    """Generate a summary for a specific document or topic."""
    try:
        qa = qa_llm()  # Use the QA model set up for this project
        
        # Check if the user wants a summary for a specific document or a topic in a document
        if 'summary of' in user_input.lower():
            if 'entire document' in user_input.lower():
                # Extract the name of the document mentioned in the request
                doc_name = user_input.split("summary of the entire document from")[-1].strip().replace("'", "")
                logging.info(f"Generating summary for the entire document: {doc_name}")
                
                # Generate a tailored prompt to summarize the entire document
                tailored_prompt = f"""
                Provide a concise summary of the entire document '{doc_name}'. Focus on the main points, policies, 
                and any key information relevant to Chartered Accountants and audits.
                """
            else:
                # Extract the section or topic the user wants summarized
                topic = user_input.split("summary of")[-1].strip().replace("'", "")
                logging.info(f"Generating summary for the topic: {topic}")
                
                # Generate a tailored prompt to summarize a specific section or topic in the document
                tailored_prompt = f"""
                Provide a summary of the section or topic '{topic}' from the provided documents. Focus on the key points, 
                policies, and any important information related to Chartered Accountants and audits.
                """
                
            # Generate the summary using the QA model
            generated_text = qa({"query": tailored_prompt})
            summary = generated_text['result']
            
            return f"Here is the summary: {summary}"
        else:
            return "Sorry, I couldn't identify a valid summary request. Please specify a document or topic for the summary."

    except Exception as e:
        logging.error(f"Error during summary generation: {str(e)}")
        raise


if __name__ == "__main__":
    data_ingestion()  # Call data ingestion on startup
