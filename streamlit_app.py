import streamlit as st 

from langchain_openai import ChatOpenAI , OpenAIEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage ## mesaj rollerini goster


from dotenv import load_dotenv 
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not found")

os.environ["OPENAI_API_KEY"] = api_key

st.set_page_config(page_title = "Ecza Destek Botu" , page_icon = ":robot_face:")
st.title("💊 İlaç ve Semptom Danışmanı")
st.write("Şikayetinizi belirtin , hemen AI asistanı şikayetinize uygun reçetesiz ilaç var ise öneride bulunsun. Türkçe desteklidir.")

loader = PyPDFLoader("recetesiz_ilac_listesi.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size = 500 , chunk_overlap = 50)
docs = splitter.split_documents(documents)

embedding = OpenAIEmbeddings(model="text-embedding-3-large")

vector_db = FAISS.from_documents(docs,embedding)

memory = ConversationBufferMemory(
    memory_key = "chat_history" , 
    return_messages=True
)


llm = ChatOpenAI(
    model_name = "gpt-4o-mini" , 
    temperature = 0.1
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm = llm , 
    retriever = vector_db.as_retriever(search_kwargs = {"k":3}) ,
    memory = memory 
)

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = qa_chain
if "chat_history" not in st.session_state: 
    st.session_state.chat_history = []

# Sayfayı iki sütuna böl: input / output
col1, col2 = st.columns([1,2]) # sag taraf biraz daha geniş

with col1: # input/sayfanin sol tarafi

    if "qa_chain" in st.session_state:
        user_question = st.text_input("👤 Şikayetinizi yazın:")
        st.button("Şikayeti Gönder")

        if user_question:
            response = st.session_state.qa_chain.invoke(user_question)
            st.session_state.chat_history.append(("👤" , user_question))
            st.session_state.chat_history.append(("🤖" , response["answer"]))
with col2: # output/sayfanin sag tarafi  

    if st.session_state.chat_history:
        st.subheader("🗨️ Sohbet Geçmişi")

        if st.button("🧹 Sohbeti Temizle"):
            st.session_state.chat_history = []
            st.session_state.qa_chain.memory.clear()
            st.rerun()

        for msg in st.session_state.qa_chain.memory.chat_memory.messages:
            if isinstance(msg,HumanMessage):
                st.markdown(f"**👤 Kullanıcı:** {msg.content}" )
            elif isinstance(msg,AIMessage):
                st.markdown(f"**🤖 Asistan:** {msg.content} ")
            else:
                st.markdown(f"**{type(msg).__name__}**: {msg.content}")
