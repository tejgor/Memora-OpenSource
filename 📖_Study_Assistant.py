import streamlit as st
from pypdf import PdfReader
import docx2txt
from langchain.vectorstores import FAISS
import src.processing as pr
import src.el_professor as ep
from PIL import Image
import io

def clear_cache():
    st.cache_data.clear()
    if 'upld_filename' in st.session_state:
        del st.session_state['upld_filename']
    if 'chunks' in st.session_state:
        del st.session_state['chunks']

def clear_answer_cache():
    answer_cache.clear()

@st.cache_data(ttl = 2*3600, max_entries = 5)
def text_process(uploads):
    text = ""
    for doc in uploads:
        if doc.name.endswith(".txt"):
            stringio = io.StringIO(doc.getvalue().decode("utf-8"))
            text += stringio.read()
        if doc.name.endswith(".pdf"):
            pdf_reader = PdfReader(doc)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        if doc.name.endswith(".docx"):
            text += docx2txt.process(doc)
    chunks = pr.text_splitter(text, docs = True)
    return chunks

@st.cache_data(ttl = 2*3600, max_entries = 5)
def create_docstore(_chunks):
    embedder = ep.embed_type_chooser(embed_type = "o", api_key=st.secrets["openai_api_key"])
    docstore = FAISS.from_documents(_chunks, embedder)
    return docstore

@st.cache_data(ttl = 2*3600, max_entries = 5)
def load_huberman():
    embedder = ep.embed_type_chooser(embed_type = "o", api_key=st.secrets["openai_api_key"])
    docstore = FAISS.load_local("./Huberman", embedder)
    return docstore

def source_cache(resp):
    sources = ep.get_sources(resp)
    return sources

@st.cache_data(ttl = 2*3600, max_entries = 5, show_spinner=False)
def answer_cache(question, _docstore):
    if model_choice:
        response = ep.answer_question(model = chat4, docstore = _docstore, question = question)
        return response
    response = ep.answer_question(model = chat3_5, docstore = _docstore, question = question)
    return response

@st.cache_data(ttl = 2*3600, max_entries = 5, show_spinner=False)
def regen(answer, detail):
    if model_choice:
        new_answer = ep.regen_answer(answer, detail, model=chat4)
        return new_answer
    new_answer = ep.regen_answer(answer, detail, model=chat3_5)
    return new_answer

# ------------------------------ Page Logic --------------------------------

logo = Image.open("./static/logo_2.png")
st.set_page_config(page_title = "Memora - Study Wise", page_icon = logo, layout="wide")
    
col1, col2= st.columns([0.5, 2])
col1.image(logo, output_format="PNG", clamp=True, use_column_width=True)

chat3_5, chat4 = pr.initialise_llms_with_key(st.secrets["openai_api_key"])

with col2:
    st.title("Memora - Study Wise")
    st.subheader("Study Assistant")
    st.markdown("##### Upload your lecture notes/slides and ask your virtual professor any questions.")
    st.markdown("""Use this tool to allow handwritten study materials to be analysed: <a href = "https://tools.pdf24.org/en/ocr-pdf">PDF OCR</a>""", unsafe_allow_html = True, help = "This tool is not affiliated with Memora")

uploaded_files = st.file_uploader("Upload Study Materials", type = ["pdf","txt","docx"], accept_multiple_files = True, on_change = clear_cache, key = "assistant_uploads")
st.session_state.upld_filename = [doc.name for doc in uploaded_files]
if 'upld_filename' in st.session_state and st.session_state.upld_filename is not None and st.session_state.upld_filename != []:
    filenames = ', '.join(st.session_state.upld_filename[:5])
    if len(st.session_state.upld_filename) > 5:
        filenames += "..."
else:
    filenames = "None"

with st.sidebar:
    model_choice = st.checkbox(":sparkle: Use GPT-4 (slower but better output)", key = "model_choice", on_change = clear_answer_cache, value = False, help = "Make sure you have access to GPT-4 API before using this option.")
    filenames_slt = st.empty()
    st.divider()
    st.caption("Special Features:")
    huberman = st.checkbox("Enable 'Chat with Dr. Huberman'", value = False, key = "huberman", help = "Enabling this will load the transcripts of 90 episodes of the Huberman Lab Podcast so you can 'ask Dr. Huberman questions'.")
    # vid = st.checkbox("LoFi music", value = False, key = "vid")
    # if vid:
    #     st.video("https://youtu.be/jfKfPfyJRdk")

if uploaded_files or huberman:
    if huberman:
        st.write("Virtual Dr. Huberman enabled!")
        docstore = load_huberman()
    else:
        chunks = text_process(uploaded_files)
        if "chunks" not in st.session_state or st.session_state.chunks == [] or st.session_state.chunks is None:
            st.session_state.chunks = chunks
        try:
            docstore = create_docstore(chunks)
        except IndexError:
            st.warning("One or more of your documents were unable to be processed. Please make sure any PDFs are searchable (**use the PDF OCR tool linked above) and try again.")
            st.stop()
    st.session_state.docstore = {"filenames" : filenames, "docstore" : docstore}

try:
    if st.session_state.docstore["docstore"] is not None:
        st.caption("Files uploaded (upload new files or refresh to replace these)")
        filenames_slt.write(f"Current Files: {st.session_state.docstore['filenames']}")
        docstore = st.session_state.docstore["docstore"]
        question = st.text_area("What would you like to know?", key = "question", max_chars = 1000, height = 100, placeholder = "You can ask me anything about the subject/module you have uploaded.\ne.g. 'Explain equation (*)' or 'What is the definition of (*)?' or even questions from past papers/problem sets.")
        answer_box = st.empty()
        if question:
            with st.spinner("Thinking..."):
                response = answer_cache(_docstore = docstore, question = question)
            answer: str = response["result"]
            answer = answer.replace(":", r"\:")
            sources = source_cache(response)
            with answer_box:
                st.write("Answer: ", answer)
                # print(answer)
            with st.expander("View sources used to generate this answer"):
                source_box = st.empty()
            regen_slider = st.empty()
            regen_button_slt = st.empty()
            detail = regen_slider.slider("Level of Detail:", min_value = 1, max_value = 10, value = 1, step = 1, key = "detail_level")
            regen_button = regen_button_slt.button("Increase Answer Detail", key = "regen", type = "primary")
            source_box.text_area(label = "sources", value = sources, height=500, label_visibility = "hidden", key = "sources")
            if regen_button:
                with st.spinner("Thinking..."):
                    new_answer = regen(answer, detail)
                with answer_box:
                    st.write("Answer:", new_answer)
                if new_answer:
                    source_box.text_area(label = "sources",value = sources, disabled = True, height = 500, label_visibility = "hidden", key = "sources_")
except:
    pass

st.divider()
st.caption("*If something stops working, refresh the page twice and try again.")

# ------------------------------ FOOTER ------------------------------ #

ft = """
<style>
a:link , a:visited{
color: ##40E0D0;  /* theme's text color hex code at 75 percent brightness*/
background-color: transparent;
text-decoration: none;
}

a:hover,  a:active {
color: #81BE83; /* theme's primary color*/
background-color: transparent;
text-decoration: underline;
}

#page-container {
  position: relative;
  min-height: 1vh;
}

footer{
    visibility:hidden;
}

.footer {
position: relative-bottom;
left: 0;
bottom: 0;
width: 100%;
background-color: transparent;
color: #FFFFF; /* theme's text color hex code at 50 percent brightness*/
text-align: left; /* you can replace 'left' with 'center' or 'right' if you want*/
}
</style>

<div id="page-container">

<div class="footer">
<p>Disclaimer: Please ensure you have permission from the copyright owner when uploading any copyrighted material</p>
<br 'style= top:3px;'>
<a style='display: inline; text-align: left;' href="https://forms.gle/gj2cuH2cS4i9UMjp8" target="_blank">Report any bugs/errors here 🐞</a>
<p>
Developed by 
<a style='display: inline; text-align: left;' href="https://www.linkedin.com/in/tg120/" target="_blank">Tejas Gorla</a> <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="Linkedin" height= "11"/>
<br 'style= top:3px;'>
<a style='display: inline; text-align: left;' href="https://www.buymeacoffee.com/holonuke" target="_blank">Buy me a snack! 🍕</a>
</p>
</div>

</div>
"""
st.write(ft, unsafe_allow_html=True)
