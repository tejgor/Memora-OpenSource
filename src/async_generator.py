import os
import time
from typing import List
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
import asyncio

async def run_chain(chain: LLMChain, chunk, num ,subject):
    text = ""
    resp = await chain.arun(chunk=chunk, subject=subject, num=num)
    text += resp
    return text

CONCURRENT_CALLS_LIMIT = 10

async def gen_concurrent(chunks, chain, subject, progress_queue):
    semaphore = asyncio.Semaphore(CONCURRENT_CALLS_LIMIT)
    processed_chunks = 0
    results = [None] * len(chunks)
    tasks = []

    async def process_chunk(chunk, index):
        nonlocal processed_chunks
        async with semaphore:
            await asyncio.sleep(2)
            result = await run_chain(chain, chunk, index, subject)
            results[index] = result
            processed_chunks += 1
            await progress_queue.put(processed_chunks)

    for i, chunk in enumerate(chunks):
        task = asyncio.create_task(process_chunk(chunk, i))
        tasks.append(task)

    await asyncio.gather(*tasks, return_exceptions=True)

    return results, processed_chunks

# async def gen_concurrent(chunks, chain, subject, progress_queue):
#     semaphore = asyncio.Semaphore(CONCURRENT_CALLS_LIMIT)
#     processed_chunks = 0
    
#     async def process_chunk(chunk):
#         nonlocal processed_chunks
#         async with semaphore:
#             await asyncio.sleep(2)
#             result, cost = await run_chain(chain, chunk, processed_chunks, subject)
#             processed_chunks += 1
#             await progress_queue.put(processed_chunks)
#             return result, cost

#     tasks = [process_chunk(chunk) for chunk in chunks]
#     results_and_costs = await asyncio.gather(*tasks)
#     results = [result for result, _ in results_and_costs]
#     costs = [cost for _, cost in results_and_costs]
    
#     return results, costs, processed_chunks


# Main loop
if __name__ == "__main__":
    import processing as pr
    import generatorGPT as gen
    load_dotenv()
    
    async def print_progress(progress_queue):
        last_processed_chunks = 0
        while True:
            processed_chunks = await progress_queue.get()
            if processed_chunks != last_processed_chunks:
                print(f"Step: {processed_chunks}")
                last_processed_chunks = processed_chunks

    async def main_loop_async(chunks, chain, subject):
        progress_queue = asyncio.Queue()
        progress_task = asyncio.create_task(print_progress(progress_queue))
        results, costs, processed_chunks = await gen_concurrent(chunks, chain , subject, progress_queue)
        progress_task.cancel()
        return results, processed_chunks, costs
    
    chat, llm = pr.initialise_llms_with_key(os.getenv("OPENAI_API_KEY"))

    foldername = input("Enter the name of the folder: ")
    chain = gen.initialise_chain_no_mem(chat)
    loaders = pr.get_pdfs(foldername)
    print("Extracting text from PDFs...")
    text = pr.extract_text_loaders(loaders)
    print("Splitting text into chunks...")
    chunks = pr.text_splitter(text)
    print(f"Generated {len(chunks)} chunks.")

    s = time.perf_counter()
    bank_a, prog = asyncio.run(main_loop_async(chunks))
    elapsed = time.perf_counter() - s
    print(f"Generated: Async {len(str(bank_a))} questions in {elapsed:0.2f} seconds.")
    bank = " ".join(bank_a).replace(". Q:", ".\n\nQ:")

    with open("bank_a.txt", "w") as f:
        f.write(gen.anki_formatter(bank))