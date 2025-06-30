import os
import os.path

import streamlit as st
import wikipedia
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.readers.wikipedia import WikipediaReader

load_dotenv()

storage_path = "./vectorstore"
WIKI_PAGES = ["Star Wars (film)", "Star Trek: The Original Series"]
# WIKI_PAGES = ["The Lord of the Rings (film series)", "Middle-earth"]

# MediaWiki requires a descriptive User-Agent; generic clients get 403 + non-JSON body.
wikipedia.set_user_agent(
    os.getenv(
        "WIKIPEDIA_USER_AGENT",
        "rag-llamaindex-wikipedia/1.0 (https://github.com/esara/rag-llamaindex-wikipedia)",
    )
)


@st.cache_resource(show_spinner="Loading knowledge base...")
def get_chat_engine():
    Settings.llm = OpenAI(temperature=0.1, model="gpt-4o-mini")

    docstore_path = os.path.join(storage_path, "docstore.json")
    if os.path.exists(docstore_path):
        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        index = load_index_from_storage(storage_context)
    else:
        loader = WikipediaReader()
        documents = loader.load_data(pages=WIKI_PAGES)
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=storage_path)

    return index.as_chat_engine(chat_mode="condense_question", verbose=True)


st.title("Ask the Wiki On Star Wars & Star Trek")
# st.title("Ask the Wiki On Middle Earth and Lord of the Rings")
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Star Wars or Star Trek!"}
#         {"role": "assistant", "content": "Ask me a question about Middle Earth or the Lord of the Rings"}
    ]

chat_engine = get_chat_engine()

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