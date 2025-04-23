from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from code_reader import code_reader
from prompts import context
from dotenv import load_dotenv

load_dotenv()   # load environment variables from .env file

llm = Ollama(model="llama3.2", request_timeout=30.0)
# llama parse is not designed to read code, but it can be used to read documentation 
parser = LlamaParse(result_type="markdown") # takes the document, pushes it to the cloud, gets parsed, and returns

file_extractor = {".pdf": parser} # whenever a pdf is found, it will be parsed
documents = SimpleDirectoryReader(input_dir="data", file_extractor=file_extractor).load_data()

embed_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents=documents, embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=llm) # can now use the vector index to question the documents

# the ability to query the pdf and get the response is given as a tool to the agent
tools = [
    QueryEngineTool(query_engine=query_engine, metadata=ToolMetadata(name="api_documentation", 
        description="this gives documentation about code for an API. Use this for reading docs for the API")),
    code_reader
]

# a different LLM for code generation
code_llm = Ollama(model="codellama")

# declare agent
agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True, context=context)

# agent would pick the right tool for the job

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    # get the response from the agent
    result = agent.query(prompt)
    print(f"Response:\n {result}")