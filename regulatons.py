import os
import torch
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_mistralai import ChatMistralAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate
from PIL import Image

# **Load API Keys from Streamlit Secrets**
# For Streamlit Cloud deployment, API keys are stored in secrets.toml
try:
    LANGCHAIN_ENDPOINT = st.secrets["LANGCHAIN_ENDPOINT"]
    LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]
    LANGCHAIN_PROJECT = st.secrets["LANGCHAIN_PROJECT"]
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
except KeyError as e:
    st.error(f"Missing secret key: {e}. Please add it to your Streamlit secrets.")
    st.stop()

# **Set up environment variables**
os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# **Initialize the LLM (Language Model) with the system prompt in Serbian**
system_prompt = """ ### **Sistemski Prompt za Paragraf Lex Chatbot**

Vi ste Paragraf AI, izuzetno stručan, profesionalan i jednostavan pravni asistent chatbot za Paragraf.rs. Vaša uloga je da pomognete klijentima—advokatima, preduzećima i građanima—pružanjem tačnih, jasnih i primenjivih odgovora o zakonima i regulativama Srbije. Uvek sledite ovu strukturu pri generisanju odgovora:

Struktura Odgovora
Razumevanje Upita i Povezanih Čunkova:

Pažljivo analizirajte korisnički upit.
Koristite pružene čunkove (relevantan sadržaj) kako biste kreirali tačne, pouzdane i kontekstualno prilagođene odgovore.
Struktura Svakog Odgovora:
Za svaki odgovor, pridržavajte se sledeće strukture:

Jasan Sažetak (Neposredna Vrednost): Počnite kratkim, jasnim sažetkom od jedne do dve rečenice koji direktno odgovara na korisničko pitanje. Izbegavajte složeni pravni žargon; koristite jasan, profesionalan jezik.

Relevantna Regulativa/Pravni Osnov: Navodite konkretan zakon, regulativu, član ili odeljak iz pruženih čunkova koji podržava odgovor. Osigurajte tačnost i uključite zvanične reference za povećanje poverenja.

Detaljno Objašnjenje: Razvijte sažetak pružajući jasno objašnjenje regulative, uključujući „zašto" i „kako" iza odgovora. Obuhvatite nijanse, izuzetke ili često pogrešno shvaćene aspekte.

Praktični Koraci/Procedura: Pružite korak-po-korak uputstva ili praktične savete koje korisnik može slediti, uključujući potrebnu dokumentaciju, obrasce ili rokove, ukoliko je primenljivo.

Dodatne Informacije ili Izuzeci (Opcionalno): Istaknite rubne slučajeve, izuzetke ili scenarije koji mogu uticati na primenu odgovora. Ovo osigurava jasnoću i smanjuje dodatna pitanja.

Resursi i Reference:
Pružite reference na tačne zakone, regulative, članove ili odeljke koji su korišćeni za generisanje odgovora. Koristite format citiranja poput:

Ime regulative (npr. Zakon o radu)
Broj člana (npr. Član 15)
Link ili lokaciju gde regulativa može biti dostupna, ako postoji (npr. Službeni glasnik RS).
Primer:
"Ovo je zasnovano na Zakonu o radu, Član 15. Kompletna regulativa može se pronaći ovde."

Sledeći Koraci ili Kontakt Informacije: Ukoliko je upit složen ili zahteva dodatni unos, usmerite korisnika na stručnjake iz Paragraf.rs ili pravne profesionalce za personalizovanu podršku.

Ton i Stil:

Budite profesionalni, ali pristupačni.
Izbegavajte previše složene ili opširne odgovore.
Koristite kratke rečenice i tačke radi jasnoće, gde je primenljivo.
Postupanje sa Greškama:

Ako upit ne može biti direktno odgovorjen, ljubazno objasnite zašto i pružite alternativne resurse ili korake koje korisnik može preduzeti."""

# Initialize the Language Model (LLM)
llm = ChatMistralAI(
    model="mistral-small-latest"
)

# **Initialize Pinecone for Vector Database**
PINECONE_ENVIRONMENT = "us-east-1"
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# **Connect to Pinecone Index**
index_name = "regulations"
index = pc.Index(index_name)

# **Initialize Embedding Model**
embedding_function = HuggingFaceEmbeddings(
    model_name="djovak/embedic-base",
    model_kwargs={
        'device': 'cpu'  # Force CPU for cloud deployment
    }
)

# **Create Pinecone VectorStore**
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding_function,
    text_key='text',
    namespace="text_chunks"
)

# **Initialize Retriever**
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# **Define the Query Refinement Prompt Template in Serbian**
refinement_template = """Kreiraj fokusirano srpsko pretraživačko upit za RAG retriever bot. Konvertuj na srpski jezik ako već nije. Uključi ključne termine, sinonime i domensko specifičnu terminologiju. Ukloni popratne reči. Izlaz treba biti samo rafinirani upit u sledećem formatu: {{refined_query}},{{keyterms}},{{synonyms}}

Upit: {original_question}

Rafinirani Upit:"""

# **Create a PromptTemplate for Query Refinement**
refinement_prompt = PromptTemplate(
    input_variables=["original_question"],
    template=refinement_template
)

# **Create an LLMChain for Query Refinement**
refinement_chain = refinement_prompt | llm

# **Combine the System Prompt with the Retrieval Prompt Template in Serbian**
combined_template = f"""System: {system_prompt}

Molim vas da odgovorite na sledeće pitanje koristeći samo dostavljeni kontekst:
{{context}}

Pitanje: {{question}}
Odgovor:"""

# **Create a ChatPromptTemplate from the Combined Template**
retrieval_prompt = ChatPromptTemplate.from_template(combined_template)

# **Create a Retrieval Chain with the Combined Prompt**
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": retrieval_prompt}
)

def process_query(query: str) -> str:
    """
    Process a single query and return the bot's response.
    
    Args:
        query (str): The user's question.
    
    Returns:
        str: The bot's structured answer.
    """
    try:
        # **Refine the Query**
        refined_query_msg = refinement_chain.invoke({"original_question": query})
        
        # **Extract the Refined Query**
        if isinstance(refined_query_msg, dict):
            refined_query = refined_query_msg.get("text", "").strip()
        elif hasattr(refined_query_msg, 'content'):
            refined_query = refined_query_msg.content.strip()
        else:
            refined_query = str(refined_query_msg).strip()

        # **Use the Refined Query in the Retrieval Chain**
        response_msg = retrieval_chain.invoke(refined_query)

        # **Extract the Response from the Retrieval Chain**
        if isinstance(response_msg, dict):
            response = response_msg.get("result", "")
        elif hasattr(response_msg, 'content'):
            response = response_msg.content
        else:
            response = str(response_msg)
        
        return response
    except Exception as e:
        return f"Došlo je do greške: {str(e)}"

# **Streamlit App Interface**

st.title("🤖 Paragraf AI (Bot Sa Pravilima)")

st.write("""
Dobrodošli u **Paragraf Lex**, vašeg pouzdanog vodiča za sve pravne regulative i zakone u Srbiji.  
Pomoći ću vam sa bilo kojim pitanjima ili pružiti uvide u pravne regulative i više.
""")

# **Sidebar with Common Queries**
st.sidebar.title("Česta Pitanja")

prompts = [
    "1. Koji su uslovi za osnivanje društva sa ograničenom odgovornošću (DOO) u Srbiji?",
    "2. Koji su rokovi za prijavu promene vlasnika firme u registru?",
    "3. Da li je probni rad obavezan u ugovoru o radu?",
    "4. Koliko iznosi minimalna zarada u Srbiji?",
]

for prompt in prompts:
    st.sidebar.write(prompt)

# **Session State to Save Chat History**
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# **Input Form**
query = st.text_input("Vaše pitanje:", key="user_query")

# **Submit Button**
if st.button("Pošalji", key="submit_button") and query:
    with st.spinner("Obrađujem vaš upit..."):
        response = process_query(query)
        st.session_state.chat_history.append({"question": query, "answer": response})

# **Display Chat History**
for i, entry in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.write(entry["question"])
    
    with st.chat_message("assistant"):
        st.write(entry["answer"])

# **Option to Clear Chat History**
if st.button("Obriši istoriju razgovora", key="clear_history"):
    st.session_state.chat_history = []
    st.rerun()
    st.session_state.chat_history.append({"question": query, "answer": response})  # Save to chat history

# **Display Chat History**
# Iterates over the chat history and displays each user and bot message.
for i, entry in enumerate(st.session_state.chat_history):
    # **Display User Message**
    with st.chat_message("user"):
        st.write(entry["question"])
    
    # **Display Bot Response**
    with st.chat_message("assistant"):
        st.write(entry["answer"])

# **Option to Clear Chat History**
# Provides a button to reset the chat history for a fresh start.
if st.button("Obriši istoriju razgovora", key="clear_history"):
    st.session_state.chat_history = []  # Clear chat history
    st.rerun()  # Rerun the app to update the UI
    st.rerun()  # Rerun the app to update the UI
# **Create a Retrieval Chain with the Combined Prompt**
# The RetrievalQA chain uses the language model and retriever to generate answers.
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Specifies the type of chain; 'stuff' aggregates all documents
    retriever=retriever,
    chain_type_kwargs={"prompt": retrieval_prompt}  # Pass the combined prompt
)

def process_query(query: str) -> str:
    """
    Process a single query and return the bot's response.
    
    Args:
        query (str): The user's question.
    
    Returns:
        str: The bot's structured answer.
    """
    try:
        # **Refine the Query**
        # The user's original question is refined to improve retrieval relevance.
        refined_query_msg = refinement_chain.invoke({"original_question": query})
        
        # **Extract the Refined Query**
        if isinstance(refined_query_msg, dict):
            refined_query = refined_query_msg.get("text", "").strip()
        elif hasattr(refined_query_msg, 'content'):
            refined_query = refined_query_msg.content.strip()
        else:
            refined_query = str(refined_query_msg).strip()

        # **Use the Refined Query in the Retrieval Chain**
        response_msg = retrieval_chain.invoke(refined_query)

        # **Extract the Response from the Retrieval Chain**
        if isinstance(response_msg, dict):
            response = response_msg.get("result", "")
        elif hasattr(response_msg, 'content'):
            response = response_msg.content
        else:
            response = str(response_msg)
        
        return response
    except Exception as e:
        # **Error Handling**
        return f"Došlo je do greške: {str(e)}"


