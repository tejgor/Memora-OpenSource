# Memora - OpenSource AI Study Assistant

### Use your own notes/documents to generate questions and answers grounded in your knowledge.

#### Open Source NotebookLM alternative

### Features:

1. **Upload all your notes/documents and ask questions on the content**
   * For example:
     - Give me all the key terms and their definitions
     - What are the key findings?
2. **Generate Question-Answer pairs (Flashcards) based on your uploaded content to test your knowledge**
3. Use these generated flashcards to test yourself and ask questions to clarify any confusion.

#### If you would like to test out the features first, go to [Memora](https://memora.page "https://memora.page") to try it out for free!

### Installation:

1. Clone this repo onto your local machine
2. Set up a python virtual env using any method (Poetry recommend)
3. Install Requirements using requirements.txt or pyproject.toml
4. Add your OpenAI API key to /.streamlit/secrets.toml.template and save this without the ".template" in the same /.streamlit/ directory, so it looks like "secrets.toml"
5. Run `streamlit run 📖_Study_Assistant.py `

### Roadmap:

* [ ] Implement GPT3.5 16k model
* [ ] Update dependencies
