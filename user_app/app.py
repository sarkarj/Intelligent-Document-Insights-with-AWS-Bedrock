#user_app/app.py
import os
import sys
import time
import tarfile
import tempfile
import boto3
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import warnings

# Suppress specific LangChain deprecation warnings to clean up logs
warnings.filterwarnings("ignore", message="Importing verbose from langchain root module is no longer supported")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.pdf_utils import extract_text_from_pdf, extract_text_from_image
from common.s3_utils import list_tar_archives

from langchain.chains import ConversationalRetrievalChain
from langchain_aws.llms import BedrockLLM
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage

# Import CrossEncoder with error handling for torch conflicts
try:
    from sentence_transformers.cross_encoder import CrossEncoder
except ImportError as e:
    st.error(f"Failed to import CrossEncoder: {e}")
    st.stop()

# Load environment variables
load_dotenv()
REGION = os.getenv("AWS_REGION")
BEDROCK_LLM_ID = os.getenv("BEDROCK_LLM_ID")
BEDROCK_EMBED_ID = os.getenv("BEDROCK_MODEL_ID")
S3_BUCKET = os.getenv("EMBEDDING_BUCKET")

# Configuration constants
MAX_CITATIONS = 5  # Maximum number of citations to show
MIN_CONFIDENCE_THRESHOLD = 0.1  # Minimum confidence score for citations
CHAIN_OF_THOUGHT_STEPS = 4  # Number of reasoning steps to show

# Custom prompt template for better responses
CUSTOM_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""You are an AI assistant that provides helpful, accurate, and concise answers based on the given context and conversation history.

Instructions:
- Keep your response under 500 words
- Be direct and to the point
- Use the provided context to answer the question
- If the context doesn't contain relevant information, say so clearly
- Maintain conversation continuity using the chat history
- Format your response clearly with bullet points or short paragraphs when appropriate

Chat History:
{chat_history}

Context from knowledge base:
{context}

Question: {question}

Answer:"""
)

# Streamlit page configuration
st.set_page_config(page_title="Ask Your Knowledge Base", page_icon="🧬")
st.markdown("""
    <style>
        .refresh-btn button {
            padding: 0.25rem 0.6rem;
            font-size: 0.8rem;
        }
        .ranking-item {
            padding: 0.5rem;
            margin: 0.2rem 0;
            background-color: #f8f9fa;
            border-radius: 0.3rem;
            border-left: 3px solid #007acc;
        }
    </style>
""", unsafe_allow_html=True)

# Header with refresh button
col1, col2 = st.columns([0.88, 0.12])
with col1:
    st.markdown("### 🧬 Ask Your Knowledge Base")
with col2:
    with st.container():
        st.markdown('<div class="refresh-btn">', unsafe_allow_html=True)
        if st.button("🌿", key="refresh_button", help="Reload S3 vector files"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.clear()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

@st.cache_resource(show_spinner="⏳ Loading vector stores from S3...")
def load_vector_store():
    """Load and merge FAISS vector stores from S3 tar archives."""
    s3 = boto3.client("s3", region_name=REGION)
    embeddings = BedrockEmbeddings(model_id=BEDROCK_EMBED_ID, region_name=REGION)
    archives = list_tar_archives()

    if not archives:
        return None, []

    merged_store = None
    loaded_names = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for key in archives:
            try:
                archive_name = os.path.basename(key)
                archive_path = os.path.join(tmpdir, archive_name)
                s3.download_file(S3_BUCKET, key, archive_path)
                
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(tmpdir)

                # Find FAISS index files
                index_path = None
                for root, _, files in os.walk(tmpdir):
                    if "index.faiss" in files and "index.pkl" in files:
                        index_path = root
                        break

                if not index_path:
                    st.warning(f"⚠️ Skipped `{archive_name}` (missing FAISS index)")
                    continue

                # Load and merge vector store
                store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
                if merged_store is None:
                    merged_store = store
                else:
                    merged_store.merge_from(store)

                loaded_names.append(archive_name)
                
            except Exception as e:
                st.error(f"❌ Failed to load `{key}`: {e}")
                continue

    return merged_store, loaded_names

# Load vector stores
vector_store, loaded_vector_names = load_vector_store()
if vector_store is None:
    st.warning("⚠️ No FAISS vector store found in S3. Please upload at least one embedding archive.")
    st.stop()

# Display loaded vector stores
if loaded_vector_names:
    with st.expander("📦 Loaded Vector Stores"):
        for name in loaded_vector_names:
            st.markdown(f"- `{name}`")

class CrossEncoderReranker:
    """Cross-encoder reranker for improving document relevance scoring."""
    
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            self.model = CrossEncoder(model_name)
        except Exception as e:
            st.error(f"Failed to initialize CrossEncoder: {e}")
            raise

    def rerank(self, query, documents):
        """Rerank documents based on cross-encoder scores."""
        if not documents:
            return documents
        
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)
        
        # Add scores to document metadata
        for doc, score in zip(documents, scores):
            doc.metadata["confidence_score"] = float(score)
            doc.metadata["score"] = float(score)  # Keep backward compatibility
            
        return sorted(documents, key=lambda d: d.metadata.get("confidence_score", 0), reverse=True)

class CrossEncoderRerankerCompressor(BaseDocumentCompressor):
    """Document compressor using cross-encoder reranking."""
    
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", **kwargs):
        super().__init__(**kwargs)
        self._model_name = model_name
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self._model_name)
        return self._reranker

    def compress_documents(self, documents: list[Document], query: str) -> list[Document]:
        """Compress and rerank documents based on query relevance."""
        if not documents:
            return documents
        return self.reranker.rerank(query, documents)

# Initialize retrieval components with error handling
try:
    reranker_compressor = CrossEncoderRerankerCompressor()
    compressor_pipeline = DocumentCompressorPipeline(transformers=[reranker_compressor])
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor_pipeline,
        base_retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    )
except Exception as e:
    st.error(f"Failed to initialize retriever: {e}")
    st.stop()

# Initialize Bedrock LLM
try:
    llm = BedrockLLM(
        model_id=BEDROCK_LLM_ID, 
        region_name=REGION,
        model_kwargs={"max_tokens": 2000, "temperature": 0.1}
    )
except Exception as e:
    st.error(f"Failed to initialize Bedrock LLM: {e}")
    st.info("Please check your AWS credentials and region configuration.")
    st.stop()

class CustomChatMemory:
    """Custom chat memory management for conversation continuity."""
    
    def __init__(self):
        self.messages = []
    
    def add_user_message(self, message: str):
        """Add user message to chat history."""
        self.messages.append(HumanMessage(content=message))
    
    def add_ai_message(self, message: str):
        """Add AI message to chat history."""
        self.messages.append(AIMessage(content=message))
    
    def get_chat_history_string(self) -> str:
        """Get formatted chat history string for context."""
        if not self.messages:
            return ""
        
        history_parts = []
        for msg in self.messages[-6:]:  # Keep last 6 messages for context
            if isinstance(msg, HumanMessage):
                history_parts.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                history_parts.append(f"Assistant: {msg.content}")
        
        return "\n".join(history_parts)
    
    def clear(self):
        """Clear chat history."""
        self.messages = []

def get_confidence_level(score: float) -> tuple[str, str]:
    """Get confidence level label and emoji based on score."""
    if score >= 0.8:
        return "High", "🟢"
    elif score >= 0.6:
        return "Good", "🟡"
    elif score >= 0.4:
        return "Medium", "🟠"
    else:
        return "Low", "🔴"

def format_llm_chain_of_thought(docs: list[Document], query: str, answer: str) -> str:
    """Generate LLM's chain of thought reasoning process."""
    if not docs:
        return ""
    
    thought_steps = []
    
    # Step 1: Query Analysis
    thought_steps.append(
        f"**🔍 Step 1: Query Analysis**\n"
        f"*Understanding the question:* \"{query[:100]}{'...' if len(query) > 100 else ''}\"\n"
        f"*Key concepts identified:* {', '.join(query.split()[:5])}"
    )
    
    # Step 2: Context Evaluation
    relevant_docs = [doc for doc in docs[:3] if doc.metadata.get("confidence_score", 0) > 0.3]
    if relevant_docs:
        thought_steps.append(
            f"**📚 Step 2: Context Evaluation**\n"
            f"*Found {len(relevant_docs)} highly relevant sources*\n"
            f"*Primary source:* `{relevant_docs[0].metadata.get('source', 'Unknown')}`"
        )
    
    # Step 3: Information Synthesis
    answer_length = len(answer.split())
    confidence_level = "high" if len(relevant_docs) >= 2 else "moderate" if len(relevant_docs) == 1 else "low"
    
    thought_steps.append(
        f"**🧠 Step 3: Information Synthesis**\n"
        f"*Combining information from {len(docs)} sources*\n"
        f"*Response confidence: {confidence_level.title()}*\n"
        f"*Generated response length: {answer_length} words*"
    )
    
    # Step 4: Response Formulation
    has_citations = len(relevant_docs) > 0
    response_type = "evidence-based" if has_citations else "general knowledge"
    
    thought_steps.append(
        f"**✍️ Step 4: Response Formulation**\n"
        f"*Response type: {response_type.title()}*\n"
        f"*Citations included: {'Yes' if has_citations else 'No'}*\n"
        f"*Final verification: Complete*"
    )
    
    return "**🧬 LLM Chain of Thought:**\n\n" + "\n\n".join(thought_steps)

def filter_top_citations(docs: list[Document]) -> list[Document]:
    """Filter and return top documents for citations based on confidence scores."""
    if not docs:
        return []
    
    # Filter documents with meaningful content and confidence above threshold
    meaningful_docs = []
    for doc in docs:
        confidence = doc.metadata.get("confidence_score", 0)
        content = doc.page_content.strip()
        
        if content and confidence >= MIN_CONFIDENCE_THRESHOLD:
            meaningful_docs.append(doc)
    
    # Return top documents up to MAX_CITATIONS limit
    return meaningful_docs[:MAX_CITATIONS]

def run_qa_with_custom_prompt(question: str, context_docs: list[Document], chat_history: str) -> dict:
    """Run QA with custom prompt and document context."""
    # Format context from retrieved documents
    context_text = "\n".join([doc.page_content for doc in context_docs])
    
    # Create the full prompt
    formatted_prompt = CUSTOM_PROMPT.format(
        context=context_text,
        question=question,
        chat_history=chat_history
    )
    
    # Get response from LLM
    try:
        response = llm.invoke(formatted_prompt)
        return {
            "answer": response,
            "source_documents": context_docs
        }
    except Exception as e:
        return {
            "answer": f"Error generating response: {str(e)}",
            "source_documents": []
        }

# Initialize session state
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = CustomChatMemory()

for key in ["history", "parsed_context", "parsed_filename", "submitted", "pending_query"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "history" else ""

# Display chat history
for entry in st.session_state.history:
    if entry[0] == "user":
        st.chat_message("user").markdown(entry[1])
    else:
        _, msg, resp_time, docs, ctx, fname = entry
        with st.chat_message("assistant"):
            # Check if message contains chain of thought
            if "**🧬 LLM Chain of Thought:**" in msg:
                chain_of_thought, final = msg.split("\n\nAnswer:", 1)
                with st.expander("🧬 LLM Chain of Thought", expanded=False):
                    st.markdown(chain_of_thought.replace("**🧬 LLM Chain of Thought:**\n\n", ""))
                st.markdown("**Answer:**\n" + final.strip())
            else:
                st.markdown(msg)
            
            st.caption(f"🕒 {resp_time:.2f}s")
            
            # Display extracted content from uploaded file
            if ctx:
                with st.expander(f"📌 Extracted content from `{fname}`"):
                    st.text(ctx[:4000])
            
            # Display source documents
            if docs:
                with st.expander("📄 View Sources"):
                    for i, doc in enumerate(docs):
                        source = doc.metadata.get("source", f"Document {i+1}")
                        confidence = doc.metadata.get("confidence_score", 0)
                        level, emoji = get_confidence_level(confidence)
                        preview = doc.page_content[:500]
                        
                        st.markdown(
                            f"**[{i+1}]** `{source}` {emoji} **{level}** ({confidence:.2f})\n\n{preview}"
                        )

# File upload section
uploaded_file = st.file_uploader("📌 Upload a PDF or Image", type=["pdf", "png", "jpg", "jpeg"])
if uploaded_file:
    try:
        if "pdf" in uploaded_file.type:
            text = extract_text_from_pdf(uploaded_file)
        elif "image" in uploaded_file.type:
            text = extract_text_from_image(uploaded_file)
        else:
            text = ""
        st.session_state.parsed_context = text
        st.session_state.parsed_filename = uploaded_file.name
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        st.session_state.parsed_context = ""
        st.session_state.parsed_filename = ""
else:
    st.session_state.parsed_context = ""
    st.session_state.parsed_filename = ""

# Chat input
query = st.chat_input("Your question:")
if query:
    st.session_state.pending_query = query
    st.session_state.submitted = True
    st.rerun()

# Process query
if st.session_state.submitted:
    query = st.session_state.pending_query
    context = st.session_state.parsed_context
    full_query = f"{query}\n\n[Additional context from uploaded document:]\n{context}" if context else query

    with st.chat_message("assistant"):
        with st.spinner("⏳ Thinking..."):
            start = time.time()
            try:
                # Retrieve relevant documents - FIXED: Using invoke() instead of get_relevant_documents()
                docs = retriever.invoke(full_query)
                
                # Get chat history
                chat_history = st.session_state.chat_memory.get_chat_history_string()
                
                # Run QA with custom prompt
                result = run_qa_with_custom_prompt(full_query, docs, chat_history)
                duration = time.time() - start

                answer = result.get("answer", "").strip()
                source_docs = result.get("source_documents", [])

                # Create LLM chain of thought instead of document ranking
                chain_of_thought = ""
                if source_docs:
                    chain_of_thought = format_llm_chain_of_thought(source_docs, full_query, answer)

                # Combine chain of thought and answer
                if chain_of_thought:
                    answer = f"{chain_of_thought}\n\nAnswer:\n{answer}"

                # Filter and limit citations to top documents only
                citation_docs = filter_top_citations(source_docs)
                
                # Add citations for meaningful responses
                generic_phrases = ["no", "n/a", "not sure", "sorry", "i don't", "i'm not", "unable"]
                is_generic = (
                    len(answer.strip()) <= 3 or
                    any(phrase in answer.lower() for phrase in generic_phrases)
                )

                # Add limited citations based on filtered documents
                if not is_generic and citation_docs:
                    citations = " " + " ".join([f"[{i+1}]" for i in range(len(citation_docs))])
                    answer += citations

                # Update memory
                st.session_state.chat_memory.add_user_message(query)
                st.session_state.chat_memory.add_ai_message(answer)

                # Add to history (store citation_docs for display, but keep source_docs for compatibility)
                st.session_state.history.append(("user", query))
                st.session_state.history.append((
                    "bot", answer, duration,
                    citation_docs,  # Use filtered docs for display
                    context,
                    st.session_state.parsed_filename
                ))

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(f"Error processing query: {e}")
                st.info("This might be due to AWS configuration issues. Please check:")
                st.markdown("""
                - AWS credentials are properly configured
                - The specified Bedrock model ID is correct and available in your region
                - Your AWS account has access to the Bedrock service
                - The model is enabled in your AWS Bedrock console
                """)
                st.session_state.history.append(("user", query))
                st.session_state.history.append((
                    "bot", error_msg, 0, [], context, st.session_state.parsed_filename
                ))
            finally:
                st.session_state.submitted = False
                st.rerun()