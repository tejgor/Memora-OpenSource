import io
import time
import streamlit as st
import asyncio
from pypdf import PdfReader
import docx2txt
from PIL import Image
import tiktoken
import src.processing as pr
import src.generatorGPT as gen
import src.async_generator as ag

def clear_cache():
    st.cache_data.clear()
    if 'upld_filename' in st.session_state:
        del st.session_state['upld_filename']
    if 'doc' in st.session_state:
        del st.session_state['doc']

@st.cache_resource
def cache_chain(_chat):
    cached_chain = gen.initialise_chain_no_mem(chat = _chat) # No memory
    # cached_chain = gen.initialise_chain(chat = _chat, llm = _llm) # With memory
    return cached_chain

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
    chunks = pr.text_splitter(text)
    return chunks

async def print_progress(progress_queue):
    last_processed_chunks = 0
    while True:
        processed_chunks = await progress_queue.get()
        if processed_chunks != last_processed_chunks:
            prog_value.text(f"Step: {processed_chunks}/{size}")
            last_processed_chunks = processed_chunks

async def main_loop_async(chunks, chain, subject):
    progress_queue = asyncio.Queue()
    progress_task = asyncio.create_task(print_progress(progress_queue))
    results, processed_chunks = await ag.gen_concurrent(chunks, chain , subject, progress_queue)
    progress_task.cancel()
    return results, processed_chunks

@st.cache_data(ttl = 2*3600, max_entries = 5)
def format_output(input_file: str) -> dict:
    qa_dict = {}
    lines = input_file.strip().split('\n')
    def parse_qa_in_line(line: str):
        if '::' in line:
            q, a = line.split('::', 1)
            q = q.strip()
            a = a.strip()
            return q, a
        return None, None
    q, a = None, None
    for line in lines:
        if '::' in line:
            q, a = parse_qa_in_line(line)
            if q and a:
                qa_dict[q] = a
        else:
            if line.startswith('Q:'):
                if q and a:  # Save previous Q-A pair if a new one starts
                    qa_dict[q] = a
                q, a = line[3:].strip(), None
            elif line.startswith('A:'):
                a = line[3:].strip()
    if q and a:  # Save the last Q-A pair
        qa_dict[q] = a
    return qa_dict

# @st.cache_data(ttl = 2*3600, max_entries = 5)
# def create_pdf(doc):
#     qa_dict = format_output(doc)
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.add_font(fname='./JuliaMono.ttf')
#     pdf.set_font("JuliaMono", size=12)
#     for q, a in qa_dict.items():
#         qa_pair = f"Q: {q}\nA: {a}\n\n"
#         pdf.write(txt = qa_pair, print_sh = True)
#     return bytes(pdf.output())

# ------------------------------ Page Logic --------------------------------

logo = Image.open("./static/logo_2.png")
st.set_page_config(page_title = "Memora - Study Wise", page_icon = logo, layout="wide")

chat3_5, chat4 = pr.initialise_llms_with_key(st.secrets["openai_api_key"])

chain = cache_chain(chat3_5)

prog = 0
bank = ""
encs = tiktoken.get_encoding('cl100k_base')

col1, col2= st.columns([0.5, 2])
col1.image(logo, output_format="PNG", clamp=True, use_column_width=True)

with col2:
    st.title("Memora - Study Wise")
    st.subheader("Flashcards Generator")
    st.markdown("##### Upload your lecture notes and generate Q&A flashcards.")
    st.markdown("###### You can study the flashcards on the 'Test Yourself' page or if you prefer Anki, you can choose the 'Format for Anki' option to download a file is formatted to be easily imported into Anki.")
    st.markdown("""Use this tool to allow handwritten study materials to be analysed: <a href = "https://tools.pdf24.org/en/ocr-pdf">PDF OCR</a>""", unsafe_allow_html = True, help = "If the tool says 0 words recognised, ignore this as it has still worked.")

uploaded_files = st.file_uploader("Upload Study Materials", type = ["pdf","txt","docx"], accept_multiple_files=True, on_change = clear_cache, key = "gen_uploads")

with st.sidebar:
    if 'upld_filename' in st.session_state and st.session_state.upld_filename is not None and st.session_state.upld_filename != []:
        st.write(f"Current file(s): {st.session_state.upld_filename}")
    with st.expander(":bulb: How to use:", expanded = True):
        st.markdown("""
                    1. Upload lecture notes/study notes for a same subject/module (multiple files allowed). *Make sure the text in the pdfs is selectable.*
                    2. Enter subject/module name.
                    3. Click 'Generate Questions'.
                    4. Wait for completion, avoid refreshing or navigating away.
                    5. Larger files may take longer to process.
                    6. If process stops, click 'Generate Questions' again without refreshing.
                    7. Download generated document to import into Anki or use on Test Yourself page.""")

if uploaded_files:
    chunks = text_process(uploaded_files)
    st.session_state["chunks"] = chunks
    st.session_state.upld_filename = [doc.name for doc in uploaded_files]

if 'chunks' in st.session_state and st.session_state.chunks is not None and st.session_state.chunks != []:
    st.caption("Files uploaded (upload new files or refresh to replace these)")
    chunks = st.session_state["chunks"]
    cost = len(encs.encode(str(chunks)))*0.000002
    size = len(chunks)
    anki_format = st.checkbox("Format Q&A pairs for Anki Import?", value = False, key = "anki_format", help = "When unchecked, the output will be a text file with questions and answers separated by a line break. Otherwise, the output will be a text file with questions and answers separated by '::' that you can use to easily import into Anki.")
    subject = st.text_input("Enter the name of the subject/module:", key="subject")
    gen_button = st.button("Generate Questions", key="gen_button", type="primary")
    if gen_button and subject:
        prog_value = st.empty()
        prog_value.text(f"Step: starting process...")
        stay_open = st.empty()
        stay_open.info("Please do not close the page until the process is complete.")
        if "doc" in st.session_state and st.session_state.doc is not None and st.session_state.doc["name"] == subject and st.session_state.doc["anki"] == anki_format:
            st.success("Questions already generated! (If you want to regenerate, rename the subject/module e.g. add a number to the end)")
            download = st.download_button("Download", st.session_state.doc["doc"], file_name = f"{st.session_state.doc['name']}.txt", key="download_button")
            if download:
                time.sleep(3)
                clear_cache()
        else:
            with st.spinner("Generating questions..."):
                try:
                    bank_gen, prog = asyncio.run(main_loop_async(chunks, chain = chain, subject = subject))
                except NameError:
                    st.error("Please provide a valid API key to use this feature.")
                try:
                    bank = " ".join(bank_gen).replace(". Q:", ".\n\nQ:")
                except TypeError:
                    filtered = list(filter(None, bank_gen))
                    bank = " ".join(filtered).replace(". Q:", ".\n\nQ:")
            stay_open.empty()
            cost += len(encs.encode(bank))*0.000002
            print(f"Cost: {round(cost, 2)+0.0011*size}")
            if anki_format:
                doc = gen.anki_formatter(bank)
            else:
                doc = bank + "\n\nGenerated by Memora - Study Wise"
            st.session_state["doc"] = {"name": subject, "doc": doc, "anki" : anki_format}
            st.success("Flashcards generated! Head to the 'Test Yourself' page to use them.")
            download = st.download_button("Download", st.session_state.doc["doc"], file_name=f"{subject}.txt", key="download_button")
            # download_pdf = st.download_button(label = "Download PDF", data = create_pdf(st.session_state.doc["doc"]), file_name=f"{subject}.pdf", mime="application/pdf", key = "pdf_download_button", help = "This feature is still in beta so the questions and answers may contain missing symbols. I recommend downloading both the text file and the pdf file.")
            if download:
                time.sleep(3)
                clear_cache()

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
