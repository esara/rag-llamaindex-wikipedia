import streamlit as st
import os
import os.path

from dotenv import load_dotenv
# Updated imports for LlamaIndex v0.10+
from llama_index.core import VectorStoreIndex, Settings
# from llama_index.llms.gemini import Gemini
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# OpenTelemetry imports
from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
from opentelemetry.instrumentation.llamaindex import LlamaIndexInstrumentor
# Traceloop Import
# from opentelemetry.instrumentation.google_generativeai import GoogleGenerativeAiInstrumentor

load_dotenv()

# Only instrument LlamaIndex once to avoid "already instrumented" error
# The TracerProvider and exporter are automatically configured by opentelemetry-instrument
if 'instrumented' not in st.session_state:
    LlamaIndexInstrumentor().instrument()
    GoogleGenAiSdkInstrumentor().instrument()
#     GoogleGenerativeAiInstrumentor().instrument()
    st.session_state.instrumented = True

storage_path = "./vectorstore"

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Configure global settings instead of using ServiceContext (deprecated)
#Settings.llm = Gemini(
Settings.llm = GoogleGenAI(
    model="gemini-2.0-flash",
    # api_key="some key",  # uses GOOGLE_API_KEY env var by default
)

# Use WikipediaReader directly - no need for download_loader
loader = WikipediaReader()
# Using specific Wikipedia page titles that definitely exist
documents = loader.load_data(pages=['Star Wars (film)', 'Star Trek: The Original Series'])
# documents = loader.load_data(pages=['The Lord of the Rings (film series)', 'https://en.wikipedia.org/wiki/Middle-earth'])

# Create index
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir=storage_path)

st.title("Ask the Wiki On Star Wars & Star Trek")
# st.title("Ask the Wiki On Middle Earth and Lord of the Rings")
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Star Wars or Star Trek!"}
#         {"role": "assistant", "content": "Ask me a question about Middle Earth or the Lord of the Rings"}
    ]

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            # Display source information if available
            if hasattr(response, 'source_nodes') and response.source_nodes:
                with st.expander("View Sources"):
                    for i, node in enumerate(response.source_nodes):
                        st.write(f"**Source {i+1}:**")
                        st.write(f"Score: {node.score:.3f}")
                        st.write(f"Content: {node.text[:500]}...")
                        st.write("---")
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)