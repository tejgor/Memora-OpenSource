import re
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
try:
    from src.processing import initialise_llms, get_pdfs, extract_text_loaders, text_splitter
except ModuleNotFoundError:
    from processing import initialise_llms, get_pdfs, extract_text_loaders, text_splitter

bank = ""
progress = 0
cost = 0

def _make_prompt_gen(type: str) -> HumanMessagePromptTemplate:
    if type == "QA":
        human_prompt = HumanMessagePromptTemplate(
            prompt = PromptTemplate(
                template= """Your task is to generate question and answer pairs based on the provided notes, delimited by triple backticks, on ```{subject}```.
                Add variation in the type of questions generated and include questions that require the student to explain a concept or idea, or questions that require the student to apply a concept or idea to a given scenario making sure that the answers are scientifically, mathematically and logically sound.
                For example, if the notes are on the subject of "calculus", you could generate a question that requires the student to explain the concept of "differentiation" or a question that requires the student to apply the concept of "differentiation" to a given scenario. 
                The questions must be testing the knowledge of the notes rather than the specific layout e.g. "What are the assumptions for equation X?" rather than "What are the assumptions for equation X from section Y?".
                It is vital that all of the information in the provided notes is captured in the generated questions and answers. Prioritize quality over quantity and make sure the answers are complete and don't stop abruptly.
                Ensure that the questions and answers are clear, relevant, and comprehensive to facilitate effective understanding of the material, and are appropriate for university students. Make sure the answers to the questions are detailed and include an explaination of any relevant keywords and ideas.
                Do not generate questions about specific figures or diagrams, instead, generate questions about the concepts and ideas that the figures and diagrams are illustrating.
                If you generate a question and the context doesn't directly provide the answer, you should generate the answer using your prior knowledge, ensuring factual and scientific accuracy.
                The output should follow the specified output format.
                ----------------------------------------
                Output Format: Do not include the (cid...) tags in the generated questions and answers.
                The provided input context is extracted text from study materials so it may contain equations and symbols that are corrupted or out of place due to errors in the extraction process. To overcome this, take time to read the input context and use your prior knowledge to rewrite the input context making sure that all of the information is accurately captured.
                e.g. if the context contains "x^2", and you are able to determine that the appropriate unicode character is "²", then the generated answer should contain "x²".
                The rewritten context should contain all the correct equations and symbols and should be free of any corrupted or out of place equations and symbols. For example, if an equation looks like "ln# = ln% + 'ln)̇", you should use your prior knowledge on the input context to rewrite it to "lnσ = lnk + mlnἐ".
                The rewritten context is only for your reference and should not be included in the output.
                Please present the output in the following format: Q: '<question>'\nA: '<answer>', where <question> is the generated question and <answer> is the generated answer to the question.
                The input context may contain out of place or corrupted characters and equations so using your prior knowledge on the content of the context, you should ensure that the output is free of any corrupted characters and equations and is readable and understandable.
                If the output contains equations, assuming you have extensive prior knowledge about said equations, you should rewrite the equation, adhering to the specified output format and making sure any missing or corrupted characters are replaced with the appropriate character.
                Also ensure each question-answer pair is separated by '\n\n' and make sure any equations are clearly formatted.
                ----------------------------------------
                Notes: ```{chunk}```""",
                
                input_variables = ["subject", "chunk"],
                )
        )
    elif type == "guide":
        human_prompt = HumanMessagePromptTemplate(
            prompt = PromptTemplate(
                template= """Task: Given a study material chunk and a specified subject or module name, both delimited by triple backticks, create a structured and comprehensive learning guide segment that contains all the essential information needed for learning the content efficiently and quickly without losing any information.
                The chunks will be sent in the order they appear in the notes, use the provided chunk number, delimited by triple backticks, to keep track of this. Ensure that the learning guide segment includes key equations, facts, and wider context, so all content is covered.
                Organize the information in a structured manner that facilitates efficient learning. In addition, provide relevant learning resources such as examples and practice questions when applicable, including correct answers and ensuring the questions are logically connected to the provided notes.
                Tailor the learning guide segment to the subject or module name provided. The segments should fit together cohesively to create a complete and efficient learning resource when combined.
                Make sure to explain what each term means in equations and include a clear explaination of any relevant keywords and ideas.
                The output should follow the specified output format.
                ----------------------------------------
                Output Format: If the output contains equations, assuming you have extensive prior knowledge about said equations, you should rewrite the equation, adhering to the specified output format.
                Format all equations to be in a printable format and make sure they are on their own line and format the equations so that they are easy to read and understand.
                DO NOT include any information that includes (cid...) tags as these are placeholders for symbols and are not useful for learning.
                Number each learning guide segment based on the chunk number.
                Generate a sensible and clear name or title for each learning guide segment and number them.
                ----------------------------------------
                Subject or module name: ```{subject}```
                ----------------------------------------
                Chunk number: ```{num}```
                ----------------------------------------
                Chunk: ```{chunk}```""",
                input_variables = ["subject", "num" ,"chunk"],
                )
        )
    elif type == "explain":
        human_prompt = HumanMessagePromptTemplate(
            prompt = PromptTemplate(
                template= """You have two tasks to complete. You can receive upto two inputs, delimited by triple backticks.
                The first input is the User Input and the second is the Input context. The User Input is optional so it may not always be provided.
                In the case that only the Input context is provided, you should just carry out your secondary task.
                In the case that the User Input and Input context are both provided, carry out your primary task.
                Make sure to adhere to the specified output format.
                ----------------
                PRIMARY TASK: Use the provided Input context to make sure you are responding with the relevant information. Then, using your prior knowledge and ensuring factual and scientific accuracy, respond to the User Input in detail and explain all ideas and concepts clearly.
                SECONDARY TASK: Using your prior knowledge on the content of the context, expand and explain the context in detail, ensuring factual and scientific accuracy and explaining all ideas and concepts clearly.
                ----------------
                OUTPUT FORMAT: Only include the final answer or explanation in the output, excluding intermediate thoughts or steps.
                Present the answer in markdown format, ensuring all equations are clearly formatted to show powers etc.
                Take time to understand your task thoroughly.
                If applicable, the primary task should come before the secondary task in the output.
                If the output contains equations and you possess extensive knowledge about them, rewrite the equation to make sure it is formatted clearly and correctly.
                Do not include the User Input in the output.
                ----------------
                User Input: ```{question}```
                ----------------
                Input context: ```{answer}```""",
                input_variables = ["question", "answer"],
                )
            # prompt = PromptTemplate(
            #     template= """PRIMARY TASK: If an input question, delimited by triple backticks, is provided, answer it clearly using your prior knowledge and use the input context to understand the context.
            #     If no input question is provided, carry out your secondary task.
            #     When an input question is provided, you should return both the direct answer to the input question as well as your secondary task.
            #     If the provided input context, delimited by triple backticks, lacks relevant information, use your knowledge to answer the input question without mentioning the absence of information.
            #     Ensure explanations for terms in equations and relevant keywords are included. Verify the mathematical soundness of any equations.
            #     Adhere to the specified output format.
            #     ----------------
            #     SECONDARY TASK: Elaborate on the input context, delimited by triple backticks, in a clear and detailed manner, ensuring it is easily understood, unambiguous, and factually accurate.
            #     Adhere to the specified output format.
            #     ----------------
            #     OUTPUT FORMAT: Only include the final answer or explanation in the output, excluding intermediate thoughts or steps.
            #     Present the answer in markdown format, ensuring all equations are clearly formatted to show powers etc.
            #     Take time to understand your task thoroughly.
            #     If the output contains equations and you possess extensive knowledge about them, rewrite the equation following the specified output format.
            #     Do not incorporate the input question in the output.
            #     ----------------
            #     Input question: ```{question}```
            #     ----------------
            #     Input context: ```{answer}```""",
            #     input_variables = ["question", "answer"],
            #     )
        )        
    else:
        raise ValueError("Invalid type has to be either 'QA', 'guide' or 'explain'")
    return human_prompt

def initialise_chain_no_mem(chat, type: str = "QA") -> LLMChain:
    human_prompt = _make_prompt_gen(type)
    chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    return chain

def regen_answer_gen(question: str, answer: str, model) -> str:
    human_prompt = _make_prompt_gen("explain")
    chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
    chain = LLMChain(llm = model, prompt = chat_prompt)
    answer = chain.run(answer = answer, question = question)
    return answer

def anki_formatter(text: str) -> str:
    auth_line = "\nGenerated by Memora - Study Wise"
    qna_list = []
    # qna_pairs = re.findall(r'Q: (.*?)\nA: (.*?)\n', text, re.DOTALL)
    qna_pairs = re.findall(r'Q: (.*?)\nA: (.*?)(?=\n\nQ: |\Z)', text, re.DOTALL)
    for q, a in qna_pairs:
        qna_list.append({"Question": q.strip(), "Answer": a.strip()})
    formatted_output = "Question".ljust(80) + "::" + "Answer\n" + ("-" * 160) + "\n"
    for qna in qna_list:
        formatted_output += qna["Question"].ljust(80) + "::" + qna["Answer"] + "\n"
    # print("Anki format complete")
    formatted_output += auth_line
    return formatted_output

def main_loop_sync(chunk, local_chain: LLMChain, bank, subject):
    bank = ""
    with get_openai_callback() as cb:
        response = local_chain.run(subject = subject, chunk = chunk)
        print("success")
        # print(f"Step: {progress}/{len(chunks)} Tokens: {cb.total_tokens}")
        cost = cb.total_cost
    bank += response
    # print(f"Cost: {cost}")
    # formatted_bank = anki_formatter(bank)
    return bank, cost

def initialise_chain_with_mem(chat, llm) -> LLMChain:
    human_prompt = HumanMessagePromptTemplate(
    prompt = PromptTemplate(
        template= """Based on the provided notes, chat history, and your own knowledge, create high-quality Question and Answer pairs that will enable efficient and thorough learning of the content in the notes. 
        Ensure that the questions and answers are clear, relevant, and comprehensive to facilitate effective understanding and learning of the material.
        It is critical that all of the information in the provided notes is captured in the generated questions and answers. Prioritize quality over quantity and make sure the answers are complete and don't stop abruptly.
        Please present the output in the following format: 'Q': '<question>', 'A': '<answer>', where <question> is the generated question and <answer> is the generated answer to the question and ensure each question-answer pair is on a new line.
        Notes: {chunk}""",
        # template = "Analyze the provided text and use the conversation history to generate insightful study questions that help understand and learn the content. Additionally, draw upon your wider knowledge of the topic to create questions that encourage deeper exploration and understanding. The output should be in the following format: 'Q' : '<question>', 'A' : '<answer>'. Where <question> is the generated question and <answer> is the generated answer to the question. Text: {chunk}",
        input_variables = ["chunk"],
        )
    )
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=300)
    chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
    chain = LLMChain(llm=chat,memory = memory, prompt=chat_prompt)
    return chain

# Main loop
if __name__ == "__main__":
    chat, llm = initialise_llms()
    foldername = input("Enter the name of the folder: ")
    chain = initialise_chain_with_mem(chat, llm)
    loaders = get_pdfs(foldername)
    print("Extracting text from PDFs...")
    text = extract_text_loaders(loaders)
    print("Splitting text into chunks...")
    chunks = text_splitter(text)
    print("Generating Q&A pairs...(this may take a while depending on the size of your document(s))")
