from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate # helpers for loading documents and creating indexes
from llama_index.core.embeddings import resolve_embed_model # embedding model
from llama_index.core.tools import QueryEngineTool, ToolMetadata 
from llama_index.core.agent import ReActAgent # agent that can read code and documentation
from pydantic import BaseModel # pydantic is used to define the output of the agent
from llama_index.core.output_parsers import PydanticOutputParser # allows us to parse the output of the agent
from llama_index.core.query_pipeline import QueryPipeline # allows us to chain multiple queries together
from prompts import code_parser_template # prompts for the agent
from code_reader import code_reader
from prompts import context
from dotenv import load_dotenv
import json
import ast
import os
import re

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
# agent would pick the right tool for the job
agent = ReActAgent.from_tools(tools=tools, llm=code_llm, verbose=True, context=context)


# llamaindex and another llm can format the output into something that follows the pydantic model
class CodeOutput(BaseModel):
    code: str
    description: str
    filename: str

parser = PydanticOutputParser(CodeOutput)
json_prompt_str = parser.format(code_parser_template)
# make sure to declare the input var here
json_prompt_tmpl = PromptTemplate(template=json_prompt_str, input_variables="response")
output_pipeline = QueryPipeline(chain=[json_prompt_tmpl, llm])

def to_dict(llm_output: str) -> dict:
    # 1) grab the {...} block
    m = re.search(r'({[\s\S]*})', llm_output)
    if not m:
        raise ValueError("No JSON-like block found.")
    js = m.group(1)
    # 2) convert any """...""" sections into one JSON string value
    js = re.sub(
        r'"""([\s\S]*?)"""',
        lambda mo: '"' + mo.group(1).replace('\n', '\\n').replace('"', '\\"') + '"',
        js,
        flags=re.DOTALL,
    )
    # 3) parse
    try:
        return json.loads(js)
    except json.JSONDecodeError:
        return ast.literal_eval(js)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    retries = 0

    while retries < 3:
        try:
            result = agent.query(prompt)
            next_result = output_pipeline.run(response=result)
            cleaned_output = str(next_result).replace("assistant:", "").strip()
            data_dict = to_dict(cleaned_output)
            break
        except Exception as e:
            retries += 1
            print(f"Error occured, retry #{retries}:", e)

    if retries >= 3:
        print("Unable to process request, try again...")
        continue

    # print and save the generated code
    print("Code generated")
    print(data_dict["code"])
    print("\n\nDescription:", data_dict["description"])

    filename = data_dict["filename"]
    try:
        with open(os.path.join("output", filename), "w") as f:
            f.write(data_dict["code"])
        print("Saved file", filename)
    except Exception:
        print("Error saving file...")