import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
load_dotenv()  # <-- Load .env automatically

from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

PDF_FILENAME = "2024.03.17.24304436v1.full.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
PERSIST_DIR = "chroma_store"


def build_vectorstore(pdf_path: Path) -> Chroma:
    """Load the PDF, split it into chunks, and store embeddings in Chroma."""
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunked_docs = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )


def build_conversational_chain(vector_store: Chroma) -> ConversationalRetrievalChain:
    """Create a conversational QA chain hooked to the Chroma retriever."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        convert_system_message_to_human=True,
    )
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 12, "lambda_mult": 0.5},
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )


def format_sources(source_documents: List) -> str:
    """Extract a readable citation for each retrieved chunk."""
    formatted = []
    for doc in source_documents:
        metadata = doc.metadata or {}
        page = metadata.get("page", "N/A")
        formatted.append(
            f"- Page {page + 1 if isinstance(page, int) else page}: {metadata.get('source', PDF_FILENAME)}"
        )
    return "\n".join(dict.fromkeys(formatted))  # preserve order while deduping


def chat_with_pdf() -> None:
    """Simple REPL loop that lets a user query the PDF."""
    pdf_path = Path(__file__).with_name(PDF_FILENAME)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Could not find PDF at {pdf_path}")

    if not os.environ.get("GOOGLE_API_KEY"):
        raise EnvironmentError("Please set the GOOGLE_API_KEY environment variable.")

    print("Loading and indexing the PDF... This may take a few seconds.")
    vector_store = build_vectorstore(pdf_path)
    qa_chain = build_conversational_chain(vector_store)
    chat_history: List[Tuple[str, str]] = []

    print("\nAsk anything about the PDF (type 'exit' to quit).")
    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        response = qa_chain(
            {
                "question": question,
                "chat_history": chat_history,
            }
        )
        answer = response["answer"]
        sources = format_sources(response.get("source_documents", []))

        print(f"\nPDF Bot: {answer}")
        if sources:
            print("\nSources:")
            print(sources)

        chat_history.append((question, answer))


if __name__ == "__main__":
    chat_with_pdf()
