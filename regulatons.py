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

# Initialize Streamlit App Customization using CSS
st.markdown("""
    <style>
        /* Set the main background color */
        .main {
            background-color: #f0f2f6;
        }
        /* Customize the appearance of Streamlit buttons */
        div.stButton > button {
            background-color: #4CAF50; /* Green background */
            color: white; /* White text */
            border-radius: 5px; /* Rounded corners */
            border: 1px solid #4CAF50; /* Green border */
        }
        /* Style for chat messages */
        .chat-message {
            padding: 1.5rem; 
            border-radius: 0.5rem; 
            margin-bottom: 1rem; 
            display: flex;
        }
        /* User message styling */
        .chat-message.user {
            background-color: #2b313e; /* Dark background for user */
        }
        /* Bot message styling */
        .chat-message.bot {
            background-color: #475063; /* Slightly lighter background for bot */
        }
        /* Avatar container */
        .chat-message .avatar {
            width: 20%; /* Allocate 20% width for avatar */
        }
        /* Avatar image styling */
        .chat-message .avatar img {
            max-width: 50px; /* Maximum width */
            max-height: 50px; /* Maximum height */
            border-radius: 50%; /* Circular avatar */
            object-fit: cover; /* Ensure image covers the container */
        }
        /* Message text styling */
        .chat-message .message {
            width: 80%; /* Allocate 80% width for message */
            padding: 0 1.5rem; /* Horizontal padding */
            color: #fff; /* White text color */
        }

        /* Sidebar customization */
        [data-testid="stSidebar"] .stSelectbox {
            background-color: skyblue !important; /* Light blue background */
            border-radius: 5px; /* Rounded corners */
            padding: 5px; /* Padding inside the select box */
        }

        /* Text input field styling */
        .stTextInput>div>div>input {
            border: 2px solid #4CAF50; /* Green border */
            border-radius: 10px; /* Rounded corners */
            padding: 10px; /* Padding inside the input */
            font-size: 18px; /* Larger font size for readability */
            width: 100%; /* Full width */
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
        }
        /* Text input label styling */
        .stTextInput label {
            font-size: 20px; /* Larger label font */
            font-weight: bold; /* Bold label text */
            color: #333; /* Dark label color for contrast */
        }

        /* Header container styling */
        .header-container {
            display: flex; /* Flex layout */
            align-items: center; /* Vertically center items */
            gap: 20px; /* Space between items */
            padding: 20px; /* Padding around the container */
        }

        /* Header text styling */
        .header-container h1 {
            font-size: 32px; /* Larger font size */
            color: #222; /* Dark text color */
            margin: 0; /* Remove default margin */
            font-family: 'Arial', sans-serif; /* Modern font */
            line-height: 1.2; /* Tight line spacing */
        }

        /* Set base theme and background color */
        body {
            background-color: #d2c5c5; /* Light background color */
            color: #000; /* Black text for contrast */
        }

        /* Customize the Streamlit container */
        .stApp {
            background-color: #d2c5c5; /* Matching background color */
        }

        /* Customize sidebar content background */
        .sidebar .sidebar-content {
            background-color: #d2c5c5; /* Light background for sidebar */
        }
    </style>
""", unsafe_allow_html=True)

# **Hardcoded Environment Variables**
# These are the API keys and endpoints required for the bot to function.
# **Security Note:** Hardcoding API keys is not recommended for production.
# Consider using environment variables or secret managers instead.
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = "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"
LANGCHAIN_PROJECT = "pr-long-replacement-23"
MISTRAL_API_KEY = "OWEMpjjsDfW4CEJ5gl1ZC52gjBOqFOu5"
PINECONE_API_KEY = "pcsk_4CJx2t_Mrw4gnAyaL5xabNm1TVzDRAo3LQAKp5McKDd2evMhjgt1BzZ6VesJwmVU5envBE"

# **Set up environment variables**
# These lines set the environment variables for use in the application.
os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# **Initialize the LLM (Language Model) with the system prompt in Serbian**
# The system prompt defines the behavior and structure of the bot's responses.
system_prompt = """ ### **Sistemski Prompt za Paragraf Lex Chatbot**

Vi ste **Paragraf Lex**, izuzetno stručan, profesionalan i jednostavan pravni asistent chatbot za Paragraf.rs. Vaša uloga je da pomognete klijentima—advokatima, preduzećima i građanima—pružanjem tačnih, jasnih i primenjivih odgovora o zakonima i regulativama Srbije. Uvek sledite ovu strukturu pri generisanju odgovora:

---

### **Struktura Odgovora**  

1. **Razumevanje Upita i Povezanih Čunkova**:  
   - Pažljivo analizirajte korisnički upit.  
   - Koristite pružene čunkove (relevantan sadržaj) kako biste kreirali tačne, pouzdane i kontekstualno prilagođene odgovore.

2. **Struktura Svakog Odgovora**:  
   Za svaki odgovor, pridržavajte se sledeće strukture:

   - **Jasan Sažetak (Neposredna Vrednost)**: Počnite kratkim, jasnim sažetkom od jedne do dve rečenice koji direktno odgovara na korisničko pitanje. Izbegavajte složeni pravni žargon; koristite jasan, profesionalan jezik.  

   - **Relevantna Regulativa/Pravni Osnov**: Navodite konkretan zakon, regulativu, član ili odeljak iz pruženih čunkova koji podržava odgovor. Osigurajte tačnost i uključite zvanične reference za povećanje poverenja.  

   - **Detaljno Objašnjenje**: Razvijte sažetak pružajući jasno objašnjenje regulative, uključujući „zašto“ i „kako“ iza odgovora. Obuhvatite nijanse, izuzetke ili često pogrešno shvaćene aspekte.  

   - **Praktični Koraci/Procedura**: Pružite korak-po-korak uputstva ili praktične savete koje korisnik može slediti, uključujući potrebnu dokumentaciju, obrasce ili rokove, ukoliko je primenljivo.  

   - **Dodatne Informacije ili Izuzeci (Opcionalno)**: Istaknite rubne slučajeve, izuzetke ili scenarije koji mogu uticati na primenu odgovora. Ovo osigurava jasnoću i smanjuje dodatna pitanja.  

   - **Primer/Scenarijo (Opcionalno, ali Efikasno)**: Ukoliko je primenljivo, uključite primer iz stvarnog života ili hipotetički scenario kako biste učinili odgovor razumljivijim i relevantnijim.  

   - **Resursi i Reference**: Uputite korisnika na dodatne resurse, kao što su zvanične web stranice, obrasci za preuzimanje ili obimne baze podataka, za dodatno istraživanje ili pomoć.  

   - **Sledeći Koraci ili Kontakt Informacije**: Ukoliko je upit složen ili zahteva dodatni unos, usmerite korisnika na stručnjake iz Paragraf.rs ili pravne profesionalce za personalizovanu podršku.  

3. **Ton i Stil**:  
   - Budite profesionalni, ali pristupačni.  
   - Izbegavajte previše složene ili opširne odgovore.  
   - Koristite kratke rečenice i tačke radi jasnoće, gde je primenljivo.  

4. **Postupanje sa Greškama**:  
   - Ako upit ne može biti direktno odgovorjen, ljubazno objasnite zašto i pružite alternativne resurse ili korake koje korisnik može preduzeti.  

5. **Primer Odgovora za Ilustraciju**:  
   **Upit**: „Da li stranci mogu kupiti poljoprivredno zemljište u Srbiji?“  
   **Odgovor**:  
   - *Sažetak*: „Stranci generalno ne mogu kupiti poljoprivredno zemljište u Srbiji, ali postoje izuzeci pod određenim uslovima.“  
   - *Relevantna Regulativa*: „Ovo je regulisano Zakonom o poljoprivrednom zemljištu, član 72.“  
   - *Objašnjenje*: „Ovo ograničenje postoji kako bi se zaštitili domaći poljoprivredni interesi. Međutim, stranci mogu steći poljoprivredno zemljište putem nasledstva ili prema posebnim sporazumima između Srbije i njihove države.“  
   - *Praktični Koraci*:  
      1. Proverite da li vaša država ima poseban sporazum sa Srbijom.  
      2. Ukoliko je primenljivo, podnesite zahtev za odobrenje Ministarstvu poljoprivrede.  
   - *Primer*: „Na primer, građanin Mađarske može kupiti zemljište ako je to dozvoljeno prema bilateralnom sporazumu između Srbije i Mađarske.“  
   - *Resursi*: „Više detalja možete pronaći na zvaničnoj web stranici Ministarstva poljoprivrede [link].“  
   - *Sledeći Koraci*: „Za pomoć, kontaktirajte stručnjake Paragraf.rs na [kontakt podaci].“

---

### **Zašto Ovo Funkcioniše**  
- **Jasnoća**: Sažetak odmah pruža odgovor.  
- **Pouzdanost**: Reference zakonima povećavaju poverenje.  
- **Praktičnost**: Koraci i resursi olakšavaju primenu.  
- **Predviđanje**: Izuzeci i primeri smanjuju dodatna pitanja.  
- **Podrška**: Kontakt opcije obezbeđuju zadovoljstvo korisnika kod složenih upita.  

Koristeći ovu strukturu, chatbot će isporučivati autoritativne, prilagođene i sveobuhvatne odgovore za sve pravne upite na srpskom jeziku.."""

# Initialize the Language Model (LLM) with the system prompt
llm = ChatMistralAI(model="mistral-large-latest", system_message=system_prompt)

# **Initialize Pinecone for Vector Database**
# Pinecone is used as the vector database to store and retrieve relevant text chunks.
PINECONE_ENVIRONMENT = "us-east-1"  # Specify your Pinecone environment
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# **Connect to Pinecone Index**
# Connect to the specific Pinecone index that contains regulations data.
index_name = "regulations"  # Name of the Pinecone index for regulations
index = pc.Index(index_name)

# **Initialize Embedding Model**
# HuggingFaceEmbeddings converts text into vectors for similarity search.
embedding_function = HuggingFaceEmbeddings(
    model_name="djovak/embedic-base",  # Specify the embedding model
    model_kwargs={
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
    }
)

# **Check for CUDA Availability**
# Inform the user if the model is running on CPU, which may be slower.
if not torch.cuda.is_available():
    st.warning("Warning: CUDA is not available. The model will run on CPU, which may lead to slower performance.")

# **Create Pinecone VectorStore**
# This integrates Pinecone with LangChain for vector-based retrieval.
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding_function,
    text_key='text',  # The key in your Pinecone index where the text is stored
    namespace="text_chunks"  # Namespace for organizing vectors
)

# **Initialize Retriever**
# The retriever fetches the top 'k' relevant text chunks for a given query.
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # Retrieve top 4 relevant chunks

# **Define the Query Refinement Prompt Template in Serbian**
# This prompt refines user queries to improve retrieval accuracy.
refinement_template = """Kreiraj fokusirano srpsko pretraživačko upit za RAG retriever bot. Konvertuj na srpski jezik ako već nije. Uključi ključne termine, sinonime i domensko specifičnu terminologiju. Ukloni popratne reči. Izlaz treba biti samo rafinirani upit u sledećem formatu: {{refined_query}},{{keyterms}},{{synonyms}}

Upit: {original_question}

Rafinirani Upit:"""

# **Create a PromptTemplate for Query Refinement**
# This template structures how the query is refined before retrieval.
refinement_prompt = PromptTemplate(
    input_variables=["original_question"],  # Define the input variable
    template=refinement_template  # Use the defined refinement template
)

# **Create an LLMChain for Query Refinement**
# Combines the refinement prompt with the language model to process queries.
refinement_chain = refinement_prompt | llm

# **Combine the System Prompt with the Retrieval Prompt Template in Serbian**
# This template structures how the bot uses the retrieved context to answer questions.
combined_template = f"""{system_prompt}

Molim vas da odgovorite na sledeće pitanje koristeći samo dostavljeni kontekst:
{{context}}

Pitanje: {{question}}
Odgovor:"""

# **Create a ChatPromptTemplate from the Combined Template**
# This prepares the prompt for the retrieval chain.
retrieval_prompt = ChatPromptTemplate.from_template(combined_template)

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

# **Streamlit App Interface**

# **Display Header**
# This creates a styled header for the chatbot interface.
st.markdown("""
    <h1 style="text-shadow: 2px 2px 5px #4CAF50; font-weight: bold; text-align: center;">
         💲Paragraf Lex Regulacija Chatbot📝
    </h1>
""", unsafe_allow_html=True)

# **Display Introduction Text**
# Provides a welcoming message and a brief description of the chatbot.
st.markdown("""
    <p style="font-size: 18px; color: #000000; line-height: 1.6; text-align: center;">
        👋 Dobrodošli u <strong>Paragraf Lex</strong>, vašeg pouzdanog vodiča za sve pravne regulative i zakone u Srbiji. <br>
        💡 Pomoći ću vam sa bilo kojim pitanjima ili pružiti uvide u 📝 pravne regulative i više.
    </p>
""", unsafe_allow_html=True)

# **Sidebar with Common Queries**
# Displays a list of common questions in the sidebar to guide users.
st.sidebar.title("Česta Pitanja")

# **Define Common Queries**
prompts = [
    "1. Da li stranci mogu kupiti poljoprivredno zemljište u Srbiji?",
    "2. Koji su koraci za pokretanje preduzeća u Srbiji?",
    "3. Kako funkcionše poreska prijava za fizička lica?",
    "4. Šta je Zakon o radu i koje su njegove osnovne odredbe?",
]

# **Display Each Common Query in the Sidebar**
for prompt in prompts:
    st.sidebar.write(prompt)

# **Session State to Save Chat History**
# Streamlit's session state is used to maintain the chat history across user interactions.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Initialize empty chat history

# **Input Form**
# Allows the user to input their question and submit it.
query = st.text_input("Vaše pitanje:")  # Text input field for user's question

# **Submit Button**
# When clicked, processes the user's query and appends the response to chat history.
if st.button("Pošalji") and query:
    response = process_query(query)  # Process the user's query
    st.session_state.chat_history.append({"question": query, "answer": response})  # Save to chat history

# **Display Chat History**
# Iterates over the chat history and displays each user and bot message.
for entry in st.session_state.chat_history:
    # **Display User Message**
    st.markdown(f'''
        <div class="chat-message user">
            <div class="avatar">
                <img src="https://th.bing.com/th/id/OIP.uDqZFTOXkEWF9PPDHLCntAHaHa?pid=ImgDet&rs=1">
            </div>
            <div class="message">{entry["question"]}</div>
        </div>
    ''', unsafe_allow_html=True)
    
    # **Display Bot Response**
    st.markdown(f'''
        <div class="chat-message bot">
            <div class="avatar">
                <img src="https://i.pinimg.com/originals/0c/67/5a/0c675a8e1061478d2b7b21b330093444.gif" style="max-height: 70px; max-width: 50px;">
            </div>
            <div class="message">{entry["answer"]}</div>
        </div>
    ''', unsafe_allow_html=True)
    
    # **Separator Between Messages**
    st.write("---")  # Horizontal line for separation

# **Option to Clear Chat History**
# Provides a button to reset the chat history for a fresh start.
if st.button("Obriši istoriju razgovora"):
    st.session_state.chat_history = []  # Clear chat history
    st.experimental_rerun()  # Rerun the app to update the UI
