---
title: DermaInsight AI
emoji: 🔬
colorFrom: indigo
colorTo: cyan
sdk: streamlit
sdk_version: "1.58.0"
python_version: "3.11"
app_file: Skincare.py
pinned: false
---

# DermaInsight AI Product Overview

## Non-Technical Overview

DermaInsight AI is an AI-powered dermatology education assistant that helps users explore skin lesion concerns using an uploaded image, a written description, or both. The app compares the user's input with visually similar cases from a curated skin lesion knowledge base and then provides a clear, structured clinical-style explanation.

The product is designed to support awareness, education, and early decision-making. It does not replace a dermatologist or provide a formal diagnosis. Instead, it helps users better understand visible skin features, compare similar examples, and decide when professional medical evaluation may be appropriate.

## What the Product Does

- Accepts a skin lesion photo, a text description, or both.
- Searches a medical image knowledge base for visually or semantically similar cases.
- Displays similar reference cases with metadata such as diagnosis label, body location, age, and sex where available.
- Generates an AI-assisted educational assessment based on the uploaded image, user context, and retrieved similar cases.
- Clearly reminds users to consult a licensed dermatologist for diagnosis and treatment decisions.

## Why It Is Helpful

Skin concerns can be stressful, and many users do not know how to describe what they are seeing. DermaInsight AI helps bridge that gap by turning visual and written input into a more understandable explanation.

The app is helpful because it:

- Makes skin lesion information easier to explore.
- Provides quick educational feedback before a clinical appointment.
- Helps users organize their observations in plain language.
- Shows similar cases instead of giving isolated AI output.
- Encourages responsible follow-up with medical professionals.

## Target Users

- People who want educational context about a visible skin concern.
- Students or learners exploring dermatology image retrieval workflows.
- Healthcare technology reviewers evaluating multimodal RAG concepts.
- Developers building AI-assisted medical education prototypes.

## Important Medical Disclaimer

DermaInsight AI is for education and informational support only. It is not a diagnostic medical device and should not be used as a substitute for professional medical care. Skin cancer and other dermatological conditions require evaluation by qualified clinicians. Users should consult a licensed dermatologist or healthcare provider for diagnosis, treatment, or urgent concerns.

## Technical Overview

DermaInsight AI is a Streamlit-based multimodal retrieval-augmented generation application. It combines image retrieval, text retrieval, vector search, and large language model reasoning to produce an educational skin lesion analysis workflow.

The core pipeline is:

1. Load a skin lesion dataset.
2. Store image paths and metadata in a local ChromaDB collection.
3. Generate multimodal embeddings using OpenCLIP.
4. Accept user input through Streamlit.
5. Query the vector database using image input, text input, or a combined image-text embedding.
6. Retrieve similar skin lesion cases.
7. Pass retrieved context and optional uploaded image data to an OpenAI vision-capable chat model.
8. Render similar cases and the final educational assessment in the UI.

## Main Technologies

- **Python**: Core application language.
- **Streamlit**: Frontend and interactive app framework.
- **Hugging Face Datasets**: Loads the skin lesion dataset.
- **ChromaDB**: Local vector database for persistent similarity search.
- **OpenCLIP**: Multimodal embedding model for image and text search.
- **LangChain**: Prompt and model orchestration utilities.
- **OpenAI GPT-4o**: Vision-capable language model used for final educational assessment.
- **Pillow**: Image loading and preprocessing.
- **python-dotenv**: Environment variable loading for API keys and configuration.

## Retrieval Approach

The app supports three retrieval modes:

- **Image-only search**: The uploaded image is embedded and matched against stored skin lesion images.
- **Text-only search**: The user's written description is embedded and used for semantic search.
- **Image + text search**: Image and text embeddings are combined to retrieve cases that match both visual appearance and user-provided context.

This approach makes the system more flexible than a text-only chatbot because it can reason over visual similarity before generating a response.

## AI Assessment Approach

The final assessment is generated using retrieved case context plus the user's uploaded image when available. The system prompt instructs the model to act as a dermatology education assistant, focus on visible image features, avoid definitive diagnosis, and include professional-care guidance.

The generated response is intended to be explanatory and cautious. It should describe observations, mention possible relevance of similar cases, and encourage medical follow-up when appropriate.

## Product Strengths

- Combines visual search and language understanding in one workflow.
- Uses retrieved examples to ground the AI response.
- Provides a professional, clean user interface.
- Supports both image-based and text-based user input.
- Uses a persistent local vector database for faster repeated use.

## Current Limitations

- The app depends on the quality, diversity, and labels of the underlying dataset.
- It is not clinically validated as a diagnostic tool.
- Similar retrieved cases do not guarantee the same condition.
- AI-generated explanations may still be incomplete or incorrect.
- Real-world medical use would require expert review, validation, privacy controls, and regulatory consideration.

## Future Improvements

- Add stronger medical disclaimers and emergency guidance.
- Include confidence indicators for retrieval quality.
- Add dermatologist-reviewed explanation templates.
- Improve dataset documentation and label transparency.
- Add audit logging for model inputs and outputs.
- Support privacy-preserving image handling.
- Add test coverage for retrieval, prompt formatting, and UI behavior.

## Summary

DermaInsight AI is a multimodal medical education prototype that helps users explore skin lesion concerns through image search, text search, similar case retrieval, and AI-generated explanation. Its value comes from making dermatology-related information more understandable while preserving an important safety boundary: the app supports education, not diagnosis.
