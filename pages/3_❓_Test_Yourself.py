import streamlit as st
from PIL import Image
from io import StringIO, BytesIO
import random
from gtts import gTTS
import src.generatorGPT as gen
import src.processing as pr

def clear_cache():
    st.cache_data.clear()
    if 'doc' in st.session_state:
        del st.session_state['doc']

def regen(question: str, answer: str) -> str:
    if model_choice:
        new_answer = gen.regen_answer_gen(question = question, answer = answer, model = chat4)
        return new_answer
    new_answer = gen.regen_answer_gen(question = question, answer = answer, model = chat3_5)
    return new_answer

@st.cache_data(ttl = 2*3600, max_entries = 5)
def format_input(input_file: str) -> dict:
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

def reset_counter_func():
    if 'counter' in st.session_state:
        st.session_state.counter = 0

def clear_explain_q():
    if 'explain_q' in st.session_state:
        del st.session_state['explain_q']

# ------------------------------ Page Logic --------------------------------

logo = Image.open("./static/logo_2.png")
st.set_page_config(page_title = "Memora - Study Wise", page_icon = logo, layout = "wide")

chat3_5, chat4 = pr.initialise_llms_with_key(st.secrets["openai_api_key"])

logo_col, appinfo_col= st.columns([0.5, 2])
logo_col.image(logo, output_format="PNG", clamp=True, use_column_width=True)

with appinfo_col:
    st.title("Memora - Study Wise")
    st.subheader("Test Yourself")
    st.text("Upload the text file that you generated in the previous page.")
    
success = st.empty()
sound_file = BytesIO()
auth_line = "Generated by Memora - Study Wise"
uploaded = False

if "doc" in st.session_state:
    uploaded_txt = st.file_uploader("Upload the generated text file.", type="txt")
    if uploaded_txt is not None:
        bytes_data = uploaded_txt.getvalue()
        # To convert to a string based IO:
        stringio = StringIO(uploaded_txt.getvalue().decode("utf-8"))
        input_file = stringio.read()
        uploaded = True
    else:
        input_file = st.session_state.doc["doc"]
        uploaded = True
else:
    uploaded_txt = st.file_uploader("Upload the generated text file.", type="txt")
    if uploaded_txt is not None:
        bytes_data = uploaded_txt.getvalue()
        # To convert to a string based IO:
        stringio = StringIO(uploaded_txt.getvalue().decode("utf-8"))
        input_file = stringio.read()
        st.session_state["doc"] = {"name" : uploaded_txt.name, "doc" : input_file, "anki" : None}
        uploaded = True

with st.sidebar:
    model_choice = st.checkbox(":sparkle: Use GPT-4 (slower but better output)", key = "model_choice_test")
    st.divider()
    tts_choice = st.checkbox(":sound: Text-to-Speech", key = "tts_choice")
    random_choice = st.checkbox(":twisted_rightwards_arrows: Randomise order of Q&A", value = False, key = "random_toggle", on_change = reset_counter_func)
    reset_counter = st.button("Reset counter", on_click = reset_counter_func)
    st.divider()
    if 'doc' in st.session_state and st.session_state.doc["name"] is not None:
        st.write("Current File: ", st.session_state.doc["name"])
    elif uploaded and uploaded_txt is not None:
        st.write(f"Current File: {uploaded_txt.name}")
    num_q = st.empty()
    # vid = st.checkbox("LoFi music", value = False, key = "vid")
    # if vid:
    #     st.video("https://youtu.be/jfKfPfyJRdk")

if 'counter' not in st.session_state:
    st.session_state.counter = 0

if uploaded:
    qa_dict = {}
    success.success("File uploaded successfully!")
    lines = input_file.splitlines()
    auth = bool(lines[-1] == auth_line)
    if not auth:
        st.warning("The file you have uploaded was not generated using Memora - Study Wise. Please upload a valid file.")
        st.stop()
    else:
        qa_dict = format_input(input_file)
        num_q.write(f"Number of questions: {len(qa_dict)}")
    success.empty()
    col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 3, 1], gap = "small")
    with col7:
        explain = st.button("Explain the answer", disabled = False, type = "primary")
    
    if random_choice:
        view_q = col3.button("View question", type = "primary", on_click = clear_explain_q)
        if view_q:
            explain = False
            st.session_state.a = None
            random_q, random_a = random.choice(list(qa_dict.items()))
            st.session_state.q = random_q
            st.session_state.a = random_a
    else:
        next_q = col5.button("Next Q", type="primary", on_click = clear_explain_q, use_container_width=True)
        prev_q = col3.button("Prev Q", type="primary", on_click = clear_explain_q, use_container_width=True)
        q, a = list(qa_dict.items())[st.session_state.counter]
        if next_q:
            st.session_state.counter += 1
            if st.session_state.counter == len(qa_dict):
                st.session_state.counter = 0
            explain = False
            st.session_state.a = None
            q, a = list(qa_dict.items())[st.session_state.counter]
        if prev_q:
            st.session_state.counter -= 1
            if st.session_state.counter <= -1:
                st.session_state.counter = 0
            explain = False
            st.session_state.a = None
            q, a = list(qa_dict.items())[st.session_state.counter]
        q_num = int(col4.text_input("Question number", key = "q_num", on_change = clear_explain_q, label_visibility = "collapsed", value = st.session_state.counter + 1))
        if q_num > len(qa_dict):
            st.session_state.counter = 0
            st.warning("Question number exceeds number of questions. Resetting to 1.")
        elif st.session_state.counter != q_num - 1:
            st.session_state.counter = q_num - 1
            q, a = list(qa_dict.items())[st.session_state.counter]
            explain = False
            st.session_state.a = None
        st.session_state.q = q
        st.session_state.a = a
    if 'q' in st.session_state and st.session_state.q is not None:
        if random_choice:
            st.subheader(f"Q: {st.session_state.q}")
        else:
            st.subheader(f"Q{st.session_state.counter + 1}: {st.session_state.q}")
        with st.expander("View Answer", expanded=False):
            st.write(f"A: {st.session_state.a}")
            if tts_choice:
                tts = gTTS(st.session_state.a, lang='en')
                tts.write_to_fp(sound_file)
                st.audio(sound_file)
    explain_q = st.text_input("Optional: Enter a question if you want a specific part of the answer to be explained...", key = "explain_q")
    if 'a' in st.session_state and st.session_state.a is not None:
        if explain or (explain_q and explain_q != "" and explain_q != " "):
            st.write(f"Explanation: {regen(question = explain_q, answer = st.session_state.a)}")
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
