# Multi-Agent Medical Diagnosis Chatbot

## Project Overview

This system combines:

- Vector Search (Pinecone) for longitudinal patient memory
- Medical Embeddings Model for semantic similarity
- Specialized Medical Agents (Cardiology, Pulmonology, Gastroenterology, Dermatology and Rheumatology)
- Interactive UI using Gradio to take input of patient symptoms, live display of agent outputs, and saving reports.

The goal is to simulate a collaborative clinical decision-making process using AI agents.

<img src="https://github.com/noopur-zambare/medical-diagnosis-chatbot/blob/main/assets/fllowchart.png?raw=true" width="700">


## Demo
https://github.com/user-attachments/assets/07762c3c-674f-410a-a9fe-60c053a619d1

## Quickstart

1. **Clone the repo:**
   ```bash
   git clone https://github.com/noopur-zambare/medical-diagnosis-chatbot.git
   cd medical-diagnose-chatbot
   ```
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Set up your API credentials:**
    - Create a file named api.env in the project root.
    - Add your Pinecone and OpenAI credentials:
    ```bash
    OPENAI_API_KEY=''
    PINECONE_API_KEY=''
    ```
4. **Run the system:** 
    ```bash
    python app.py
    ```

## Tech Stack Used
- Langchain
- Pinecone
- Gradio

## Project Structure
```
├── Agents.py               # Medical Agents
├── assets/                 # Demo video and workflow images
├── app.py                  # Gradio application
├── embedding_model.py      # Code to download embedding model
├── keyword.json            # Keywords for routing agents
└── vector_store.py         # Vector Database
```
