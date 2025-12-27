A semantic book recommendation system powered by Large Language Models (LLMs) that understands book descriptions, user preferences, and context to provide meaningful and personalized book recommendations. Unlike traditional recommenders based on ratings or metadata, this system leverages semantic embeddings to capture deeper relationships between books.



🚀 Features

🔍 Semantic Understanding – Uses embeddings to capture meaning from book descriptions.

🤖 LLM-powered Recommendations – Combines embeddings with LLM reasoning for better suggestions.

📈 Context-aware Personalization – Recommends books based on themes, genres, or user input queries.

⚡ Efficient Retrieval – Supports similarity search using FAISS / Pinecone / ChromaDB.

🛠️ Modular Design – Easy to extend with new datasets or recommendation strategies.

🏗️ Tech Stack

Python 3.9+

LLM / Embeddings – OpenAI, Hugging Face (SentenceTransformers)

Vector Database – FAISS / Pinecone / ChromaDB

Data Handling – Pandas, Numpy

API / App – FastAPI / Flask (for serving recommendations)

Frontend (optional) – Streamlit / React for user interaction



python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows

