

# Aegis Responder: Automated RAG FAQ Assistant

**Live Demo:** [https://veliscoagent-aegis.hf.space/](https://veliscoagent-aegis.hf.space/)

First off, let me introduce what we do at **Aegis Security Solutions**. We are dedicated to making security and compliance as painless as possible. We handle the heavy lifting of security policies and infrastructure so our clients can focus on building their products rather than constantly answering audits to prove they are secure. 

But dealing with endless security questionnaires from vendors is incredibly tedious and takes up way too much time. So, I decided to automate it.

## What I Built

I built a headless AI compliance assistant that automates answering these endless vendor security questionnaires. 

### Accessing the App
To keep the data secure, the app requires a login. You can create your own account on the landing page or use the pre-configured test account

### Using the Tool (CSV Only)
Currently, the system is optimized for **CSV files only**. To ensure the AI parses your questionnaire correctly, your file must include these two columns:
1. `Question_ID`: A unique identifier for each row (e.g., Q1, Q2).
2. `Question`: The actual text of the security or compliance question.

Once uploaded, the system searches our internal security policy documents using Retrieval-Augmented Generation (RAG) and returns exactly what you need to tell the auditors—complete with confidence scores, citations, and exact snippets from the text. 

Most importantly, if the model can't find an answer in the docs, it physically refuses to guess or hallucinate. Instead, it actively throws a "Flagged" status so I know exactly where a human needs to step in.

### How I Built It

I wanted to keep the architecture lightweight, fast, and secure:
- **Backend:** Python + FastAPI for high-performance API routing.
- **LLM:** I used the **Groq API** (running GPT OSS 120B / open-source models) for lightning-fast token generation with zero local GPU overhead.
- **Embeddings:** I went with local **SentenceTransformers** (`all-MiniLM-L6-v2`). This provides infinitely scalable vector embeddings without relying on paid APIs.
- **Databases:** Raw **SQLite** for application data and an Ephemeral **ChromaDB** instance for vector memory, allowing us to bypass heavy server setups.

### Why I Made These Choices (The Trade-offs)

- **Zero-GPU Local Embeddings:** By keeping embeddings local instead of using an API like OpenAI, I ensure we aren't sending our secure corporate documents over the internet. Data privacy is priority number one. The trade-off is slightly less powerful semantic clustering than OpenAI's `text-embedding-3`, but the strict privacy and zero-cost nature make it the right choice for this use case.
- **Ephemeral Vector DB:** I chose to rebuild the vector index completely from scratch whenever the server boots. With our current library of text documents, this takes about 1.5 seconds. The trade-off is that we can't scale to an 8,000-page library without eventually migrating to a persistent ChromaDB volume or Pinecone.
- **SQLite vs Postgres:** I used a local SQLite file instead of spinning up a full Postgres instance. It's perfect for a lightweight deployment and lightning fast, though it limits how many concurrent horizontal connections we can handle in the future.

### Future Improvements

- **Document Management UI:** Right now, updating our knowledge base requires merging a `.txt` file into the GitHub repository. My next goal is to build a solid frontend UI for admins to upload and manage documents dynamically.
- **Multimodal & PDF Support:** Security policies are often locked in complex PDFs containing architecture diagrams and tables. I plan to add multimodal pipeline support to natively ingest, extract, and reason over images, infographics, and complex PDF layouts.
- **Streaming UI via SSE:** For huge, 500-question CSVs (like the SIG Core questionnaire), the user currently stares at a loading spinner for 30 seconds. I plan to convert the frontend to use Server-Sent Events (SSE) so the table populates in real-time as answers stream in. 
- **Auto-CSV Parsing:** Our CSV importer is currently rigid (expecting exact `Question_ID` and `Question` headers). A massive win would be routing the file through an LLM first to auto-detect the targeted columns from any messy Excel sheet a client sends us.
