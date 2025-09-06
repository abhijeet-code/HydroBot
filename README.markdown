# Hydroponics Education Chatbot

## Overview
The Hydroponics Education Chatbot is an AI-powered educational tool designed to provide accessible, accurate, and beginner-friendly information about hydroponic farming. Utilizing a Retrieval-Augmented Generation (RAG) architecture, it combines Llama 2 (7B-chat) with a custom dataset of 85 PDF files to answer queries on hydroponic systems, plant selection, nutrient management, and more. Built on a Kaggle notebook, this project leverages open-source libraries and a user-friendly Gradio interface, making it ideal for hobbyists, students, and enthusiasts interested in soilless farming.

## Features
- **Natural Language Q&A**: Responds to queries like “What are the best plants for hydroponics?” or “How do I clean a hydroponic system?” with clear, concise answers.
- **Custom Knowledge Base**: Powered by a dataset of 85 PDF files, including hydroponics guides, research papers, and tutorials.
- **RAG Architecture**: Integrates FAISS vector store for efficient retrieval and Llama 2 for context-aware response generation.
- **User-Friendly Interface**: Offers an interactive web-based UI via Gradio for seamless user interaction.
- **Offline Capability**: Supports saving the FAISS index and model for offline use (with adequate compute resources).
- **Educational Focus**: Provides beginner-friendly explanations to promote sustainable farming practices.

## Tech Stack
- **Programming Language**: Python 3.8+
- **Machine Learning**:
  - Llama 2 (7B-chat) from Hugging Face for natural language generation
  - Sentence Transformers (`all-MiniLM-L6-v2`) for text embeddings
- **Libraries**:
  - LangChain: For RAG pipeline and document processing
  - FAISS (CPU): For efficient vector storage and retrieval
  - PyMuPDF: For high-quality PDF text extraction
  - Transformers: For loading and running Llama 2
  - Gradio: For the web-based chatbot interface
- **Platform**: Kaggle Notebook (GPU T4 x2 for model inference)
- **Dataset**: Custom collection of 85 PDF files on hydroponic farming, hosted on Kaggle

## Project Structure
```
hydroponics-chatbot/
├── data/
│   └── your-dataset-name/           # Kaggle dataset with 85 PDFs
├── faiss_index/                    # Saved FAISS vector store
├── fine_tuned_hydroponics_model/   # (Optional) Fine-tuned Llama 2 model
├── notebook.ipynb                   # Kaggle notebook with full code
├── requirements.txt                # Dependencies
└── README.md                       # Project documentation
```

## Getting Started

### Prerequisites
- **Kaggle Account**: Required to access the notebook and dataset.
- **Hugging Face Account**: Needed to access Llama 2 (`meta-llama/Llama-2-7b-chat-hf`).
- **Hardware**: Kaggle GPU T4 x2 or equivalent (16GB+ RAM, GPU recommended for inference).
- **Dependencies**: Install via `requirements.txt` or directly in the notebook:
  ```bash
  pip install langchain langchain-community langchain-huggingface transformers sentence-transformers faiss-cpu pymupdf gradio
  ```

### Setup Instructions
1. **Access the Notebook**:
   - Open the Kaggle notebook or clone the repository (if hosted on GitHub).
   - Add the dataset: In Kaggle, go to “Add Data” and import `your-dataset-name`.

2. **Configure Environment**:
   - Enable GPU in Kaggle: Settings > Accelerator > GPU T4 x2.
   - Enable internet access for pip installs and model downloads.

3. **Run the Notebook**:
   - Execute cells sequentially to:
     - Install dependencies.
     - Load and process PDFs.
     - Build the FAISS vector store.
     - Configure the RAG pipeline with Llama 2.
     - Launch the Gradio interface.

4. **Access the Chatbot**:
   - Run the Gradio cell to start the web UI.
   - Alternatively, use the console loop for testing:
     ```python
     while True:
         query = input("Ask about hydroponics (or 'exit'): ")
         if query.lower() == 'exit':
             break
         result = qa_chain({"query": query})
         answer = result["result"].split("Answer:")[-1].strip()
         print("Answer:", answer)
     ```

## Example Usage
**Query**: “What are the best plants for hydroponics?”  
**Response**:  
The best plants for hydroponics include lettuce, celery, basil, parsley, oregano, rosemary, sage, tarragon, thyme, strawberries, and tomatoes due to their small root systems and space requirements. Large plants like watermelon, melons, squash, pumpkin, and corn are less practical.

**Query**: “How do I clean a hydroponic system?”  
**Response**:  
1. Drain the system: Remove all water and nutrient solution from the reservoir and channels.  
2. Disassemble components: Take apart removable parts like tubes, pumps, and trays.  
3. Rinse with clean water: Flush the system to remove debris and residue.  
4. Clean with a mild disinfectant: Use a diluted hydrogen peroxide or vinegar solution to scrub components.  
5. Rinse thoroughly: Ensure no cleaning agents remain.  
6. Inspect for damage: Check for clogs or wear in tubes and pumps.  
7. Reassemble and refill: Reassemble the system and add fresh water and nutrients.

## Implementation Details
- **Data Processing**:
  - PDFs are loaded using `PyMuPDFLoader` for high-quality text extraction.
  - Text is chunked into ~1000-character segments with 200-character overlap using `RecursiveCharacterTextSplitter`.
- **Vector Store**:
  - Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2`.
  - Stored in a FAISS index (CPU) for efficient similarity search.
- **RAG Pipeline**:
  - Llama 2 (7B-chat) generates answers using the top-8 retrieved documents.
  - Custom prompt ensures concise, answer-only output:
    ```
    Based on the following context about hydroponic farming, provide a clear, concise, and complete step-by-step procedure to answer the question. Return only the answer, listing all steps in order, without including the context, instructions, or any other text.
    ```
- **Consistency Fixes**:
  - Adjusted Llama 2 parameters: `temperature=0.3`, `top_p=0.9`.
  - Post-processing removes residual prompt text (e.g., “Answer:”).
- **Interface**:
  - Gradio provides a web-based UI for user queries.
  - Console loop available for lightweight testing.

## Challenges and Solutions
- **PDF Text Extraction**: Encoding issues (e.g., “culƟvaƟon”) were resolved using `PyMuPDFLoader`.
- **Inconsistent Answers**: Addressed by lowering temperature, increasing retrieved documents (`k=8`), and using a strict prompt.
- **Kaggle Limitations**: GPU memory constraints mitigated with 8-bit quantization (`load_in_8bit=True`) and limiting `max_new_tokens`.

## Future Improvements
- **Fine-Tuning**: Fine-tune Llama 2 on a Q&A dataset derived from the PDFs for offline use without RAG.
- **Dataset Expansion**: Incorporate additional PDFs or synthetic Q&A pairs to address content gaps.
- **Deployment**: Host on Hugging Face Spaces or Streamlit for public access.
- **Multimodal Support**: Add image analysis for plant health diagnostics using datasets like PlantVillage.
- **Advanced Retrieval**: Implement hybrid search (semantic + keyword) for better handling of technical terms.

## Dataset
- **Source**: Custom dataset of 85 PDFs hosted on Kaggle.
- **Content**: Hydroponics guides, research papers, and tutorials.
- **Access**: Available via Kaggle dataset (`/kaggle/input/your-dataset-name`).

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository (if hosted on GitHub).
2. Add new PDFs, enhance the RAG pipeline, or improve the UI.
3. Submit a pull request with a clear description of changes.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments
- **Hugging Face**: For Llama 2 and sentence transformers.
- **LangChain**: For the RAG framework and document processing tools.
- **Kaggle**: For hosting the dataset and notebook environment.

## Contact
For questions or feedback, reach out via [your-email@example.com](mailto:your-email@example.com) or open an issue on the repository.