from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from dotenv import load_dotenv

load_dotenv()   # load environment variables from .env file

llm = Ollama(model="mistral", request_timeout=30.0)
parser = LlamaParse(result_type="markdown") # takes the document, pushes it to the cloud, gets parsed, and returns

file_extractor = {".pdf": parser} # whenever a pdf is found, it will be parsed
documents = SimpleDirectoryReader(input_dir="data", file_extractor=file_extractor).load_data()

embed_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents=documents, embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=llm) # can now use the vector index to question the documents

print(query_engine.query("What are some of the routes in the api?"))

