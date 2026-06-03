# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Multimodal RAG (Retrieval-Augmented Generation) system** that combines vision and language models with vector database search to provide intelligent recommendations. The system is designed to retrieve images based on text queries and then use those images as context for generating recommendations via GPT-4 Vision.

**Primary Use Case**: A flower arrangement recommendation service that accepts user text queries, retrieves relevant flower images from a vector database, and generates personalized bouquet arrangement suggestions using the images as visual context.

## Architecture & Data Flow

The system has three main flows:

### 1. Basic Multimodal RAG Flow (multimodal_rag_final.py)
```
User Query (text)
    ↓
Query ChromaDB Vector Database
    ↓
Retrieve Top 2 Matching Images (using CLIP embeddings)
    ↓
Encode Images to Base64
    ↓
Send Text + Images to GPT-4 Vision Model
    ↓
Generate Conversational Recommendations
    ↓
Display Images + AI Response
```

### 2. Setup & Initialization Flow (multimodal_start.py)
- Loads images from local filesystem or HuggingFace datasets
- Creates ChromaDB persistent vector database
- Uses OpenCLIP embeddings for multimodal understanding
- Stores images with metadata (category, item_name, etc.)
- Supports both `.add()` (new records) and `.update()` (existing records)

### 3. Streamlit Web UI Flow (multimodal_rag_final_ui.py)
- Interactive web interface for the multimodal RAG system
- Caches the vision model and dataset loading for performance
- Real-time image retrieval and suggestion generation
- Visual display of retrieved images alongside AI recommendations

## Key Components

### Vector Database (ChromaDB)
- **Purpose**: Stores image embeddings for semantic search
- **Embedding Function**: OpenCLIP (multimodal embeddings that understand both text and images)
- **Storage**: Persistent SQLite-based database at `./data/flower.db`
- **Features**: 
  - Image loader for automatic image encoding
  - Metadata support for filtering and tagging
  - Efficient similarity search

### Vision Model (GPT-4O via LangChain)
- **Model**: `gpt-4o` (multimodal model with vision capabilities)
- **Role**: Analyzes retrieved images in context of user query
- **Integration**: Via `langchain_openai.ChatOpenAI`
- **Prompt Template**: System prompt defines the model as a "talented florist" with specific output requirements (conversational tone, markdown formatting)

### LangChain Chain Architecture
The system uses LangChain's pipe operator syntax:
```python
vision_chain = image_prompt | vision_model | parser
```
This creates a composable pipeline: ChatPromptTemplate → ChatOpenAI → StrOutputParser

## Setup & Running

### Prerequisites
```bash
pip install chromadb langchain-openai langchain-core langchain-community datasets streamlit python-dotenv pillow matplotlib
```

### Environment Variables
Create a `.env` file in the root directory with:
```
OPENAI_API_KEY=your_key_here
```

### Running the Console Version
```bash
# Initial setup: Download flower dataset and populate vector database
python multimodal_rag_final.py

# After setup, run queries against the database
python multimodal_rag_final.py
```

### Running the Web UI
```bash
streamlit run multimodal_rag_final_ui.py
```
This launches a local web server (default: http://localhost:8501) with an interactive interface.

### Initial Dataset Setup
The flower dataset (102 categories, 500+ images) is downloaded from HuggingFace:
- Dataset: `huggan/flowers-102-categories`
- Images are saved to `./dataset/flowers-102-categories/`
- Vector database stores embeddings at `./data/flower.db`
- First-time setup takes several minutes due to embedding generation

## File Structure

- **multimodal_start.py**: Dataset loading, database initialization, and basic querying logic
- **multimodal_rag_final.py**: Full RAG pipeline with image encoding and GPT-4 Vision integration (console version)
- **multimodal_rag_final_ui.py**: Streamlit web UI wrapper around the RAG pipeline
- **images/**: Sample images for testing (lion, tiger, food items)
- **data/**: Vector database storage (ChromaDB persistent files)
- **dataset/**: Downloaded flower images from HuggingFace

## Important Implementation Details

### Image Encoding for LLM
Images must be encoded as base64 strings before sending to GPT-4 Vision:
```python
with open(image_path, "rb") as f:
    image_data = f.read()
encoded = base64.b64encode(image_data).decode("utf-8")
# Then inject into prompt as: "data:image/jpeg;base64,{encoded}"
```

### Multimodal Prompt Structure
The ChatPromptTemplate uses a list-based message format to support mixed content types:
```python
{
    "type": "text",
    "text": "user query here"
},
{
    "type": "image_url",
    "image_url": "data:image/jpeg;base64,..."
}
```

### ChromaDB Collection Management
- Collections are created with `get_or_create_collection()` for idempotency
- Image loading requires `ImageLoader()` initialized before collection creation
- Query results include: ids, distances, metadatas, documents, uris, data
- Filter by metadata using the `where` parameter in queries

### Caching in Streamlit UI
- `@st.cache_data`: Caches immutable data (dataset loading)
- `@st.cache_resource`: Caches expensive computations (vision model initialization)

## Development Considerations

- **API Costs**: Every query invokes GPT-4 Vision, which has associated costs. Test with small result sets and cached models.
- **Image Quality**: OpenCLIP embeddings work best with clear, well-lit images. Quality of embeddings directly affects retrieval accuracy.
- **First-Time Setup**: Initial image embedding for 500 images takes several minutes. Comment out `collection.add()` calls after first successful run to avoid re-embedding.
- **Metadata Filtering**: Optional metadata (category, item_name, etc.) can filter results before LLM processing to improve relevance.

## License

MIT License (2024 Packt)
