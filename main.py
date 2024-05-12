# Imports
import google.generativeai as genai
import google.ai.generativelanguage as glm
from prompts import *
from google.oauth2 import service_account
import os
import dotenv
from langchain.text_splitter import CharacterTextSplitter
import asyncio


#   Setup
service_account_file_name = 'service_account_key.json'  #   JSON file (hidden) to access corpus

credentials = service_account.Credentials.from_service_account_file(service_account_file_name)    #   Initialising it

scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/generative-language.retriever'])

#   Setting up clients to be accesses
generative_service_client = glm.GenerativeServiceClient(credentials=scoped_credentials)   
retriever_service_client = glm.RetrieverServiceClient(credentials=scoped_credentials)
permission_service_client = glm.PermissionServiceClient(credentials=scoped_credentials)


#   Deleting a corpus
def delete_corpus(document_resource_name):
  req = glm.DeleteCorpusRequest(name=document_resource_name, force=True)
  delete_corpus_response = retriever_service_client.delete_corpus(req)
  return delete_corpus_response


#   Deleting a corpus
def delete_document(corpus_resource_name):
  req = glm.DeleteDocumentRequest(name=corpus_resource_name, force=True)
  delete_document_response = retriever_service_client.delete_document(req)
  return delete_document_response


#   Creating a corpus
def create_corpus(display_name_corpus):
  corpus = glm.Corpus(display_name=display_name_corpus)
  create_corpus_request = glm.CreateCorpusRequest(corpus=corpus)

  # Make the request
  create_corpus_response = retriever_service_client.create_corpus(create_corpus_request)

  # Set the `corpus_resource_name` for subsequent sections.
  corpus_resource_name = create_corpus_response.name
  print(create_corpus_response)

  return corpus_resource_name


#   Requesting a corpus
def get_request_corpus(corpus_name):
  get_corpus_request = glm.GetCorpusRequest(name=corpus_name)

  # Make the request
  get_corpus_response = retriever_service_client.get_corpus(get_corpus_request)

  # Print the response
  return get_corpus_response


#   Creating a document
def corpus_create_document(display_name_document, corpus_resource_name):
  created_document = glm.Document(display_name=display_name_document)

  create_document_request = glm.CreateDocumentRequest(parent=corpus_resource_name, document=created_document)
  create_document_response = retriever_service_client.create_document(create_document_request)

  # Set the `document_resource_name` for subsequent sections.
  document_resource_name = create_document_response.name
  print(create_document_response)

  return document_resource_name


#   Requesting a document
def corpus_get_request_document(document_resource_name):
  get_document_request = glm.GetDocumentRequest(name=document_resource_name)

  # Make the request
  # document_resource_name is a variable set in the "Create a document" section.
  get_document_response = retriever_service_client.get_document(get_document_request)

  # Print the response
  return get_document_response


#   Chunk splitter for optimal data retrieving and usage
def chunk_splitter(document_resource_name, document, chunk_size=2530, chunk_overlap=20):
  text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size = chunk_size,
    chunk_overlap  = chunk_overlap
  )

  passages = text_splitter.create_documents([document])


  chunks = []
  for passage in passages:
      
      chunk = glm.Chunk(data={"string_value": str(passage)})
      
      chunks.append(chunk)

  print(len(chunks))


  create_chunk_requests = []
  for element in chunks:
    create_chunk_requests.append(glm.CreateChunkRequest(parent=document_resource_name, chunk=element))


  #   This code in this function from here has to be manually changed until a found solution, depending from document to document. 100 chunks per batch max
  request = glm.BatchCreateChunksRequest(parent=document_resource_name, requests=create_chunk_requests[0:99])
  request2 = glm.BatchCreateChunksRequest(parent=document_resource_name, requests=create_chunk_requests[100:199])
  request3 = glm.BatchCreateChunksRequest(parent=document_resource_name, requests=create_chunk_requests[200:299])
  request4 = glm.BatchCreateChunksRequest(parent=document_resource_name, requests=create_chunk_requests[300:399])
  request5 = glm.BatchCreateChunksRequest(parent=document_resource_name, requests=create_chunk_requests[400:499])
  request6 = glm.BatchCreateChunksRequest(parent=document_resource_name, requests=create_chunk_requests[500:599])
  request7 = glm.BatchCreateChunksRequest(parent=document_resource_name, requests=create_chunk_requests[600:-1])

  response = retriever_service_client.batch_create_chunks(request)
  response2 = retriever_service_client.batch_create_chunks(request2)
  response3 = retriever_service_client.batch_create_chunks(request3)
  response4 = retriever_service_client.batch_create_chunks(request4)
  response5 = retriever_service_client.batch_create_chunks(request5)
  response6 = retriever_service_client.batch_create_chunks(request6)
  response7 = retriever_service_client.batch_create_chunks(request7)

  return response.chunks, response2.chunks, response3.chunks, response4.chunks, response5.chunks, response6.chunks, response7.chunks


#   Querying the corpus to recieve the relevant chunks to a query
def corpus_query(user_query, results_count, corpus_resource_name):
  
  request = glm.QueryCorpusRequest(name=corpus_resource_name, query=user_query, results_count=results_count)
  query_corpus_response = retriever_service_client.query_corpus(request)

  return query_corpus_response


#   Querying the document to recieve the relevant chunks to a query
def document_query(user_query, results_count, document_resource_name):
  
  request = glm.QueryDocumentRequest(name=document_resource_name, query=user_query, results_count=results_count)
  query_document_response = retriever_service_client.query_document(request)

  return query_document_response


#   Setup prebuilt to quickly go through process
def prebuilt_setup(display_name_corpus, display_name_document, document, chunk_size, chunk_overlap):
  corpus_resource_name = create_corpus(display_name_corpus)
  document_resource_name = corpus_create_document(display_name_document, corpus_resource_name)
  chunks = chunk_splitter(document_resource_name, document, chunk_size, chunk_overlap)

  return corpus_resource_name, document_resource_name, chunks





class UserQuery:

  def __init__(self):

    #   Settings to be applied for safety
    self.safety_settings = [
    {
      "category": "HARM_CATEGORY_HARASSMENT",
      "threshold": "HARM_BLOCK_THRESHOLD_UNSPECIFIED"
    },
    {
      "category": "HARM_CATEGORY_HATE_SPEECH",
      "threshold": "HARM_BLOCK_THRESHOLD_UNSPECIFIED"
    },
    {
      "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
      "threshold": "HARM_BLOCK_THRESHOLD_UNSPECIFIED"
    },
    {
      "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
      "threshold": "HARM_BLOCK_THRESHOLD_UNSPECIFIED"
    }
    ]

    self.generation_config = {
      "temperature": 0.7,
      "top_p": 0.9,
      "top_k": 15,
    }

    self.model = genai.GenerativeModel(model_name = "gemini-1.5-pro-latest", safety_settings = self.safety_settings, generation_config = self.generation_config)


  async def user_query(self, user_input, explanation_description, relevant_chunk_count):

    chat_messages = []  #   History of chat messages

    #   Querying document
    document_request = document_query(user_input, relevant_chunk_count, "corpora/environmentalhealthai-56dtvvug606r/documents/environmenthealthaihackatho-upwnfs1zk9ew")

    #   Saving all chunks in one string
    total_chunks = ""
    
    #   Iterating over chunks
    for chunk_iter in range(relevant_chunk_count):
      chunk = document_request.relevant_chunks[chunk_iter].chunk.data.string_value
      total_chunks += f"\n\nChunk {chunk_iter + 1}: \n\n" + chunk.strip('page_content=')


    #   Adding to chat messages the user query prompt
    chat_messages.append({"role": "user", "parts": [professional_persona_prompt_user_query_func(), data_chunks_prompt_user_query_func(total_chunks), user_prompt_func(user_input, explanation_description)]})    #   Appending teacher persona, document, and user query prompts for a structured respone to be generated whilst creating for a better context of the chat to be provided to Gemini Pro
  
    response = self.model.generate_content(chat_messages)    #   Passing chat to Gemini Pro for generation

    try:
      chat_messages.append({"role": "model", "parts": [response.parts[0].text]})    #   Saving the model's response as reference guide of responding
    
    except IndexError:
      while len(response.parts) == 0:
        response = self.model.generate_content(chat_messages)    #   Passing chat to Gemini Pro for generation

        print("IndexError bypassed")
        
      chat_messages.append({"role": "model", "parts": [response.parts[0].text]})    #   Saving the model's response as reference guide of responding

    return response.parts[0].text



if __name__ == "__main__":

  explanation_description = input("Explain how you want to your answer to be answered like (e.g. a 5 year old playing with toys): ")

  corpus_resource_name = "corpora/environmentalhealthai-56dtvvug606r"

  dotenv.load_dotenv()    #   Loading the .env file
  genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))    #   Configuring the API Key

  relevant_chunk_count = 10

  while True:
    user_query = input("Your question: ")

    document_request = document_query(user_query, relevant_chunk_count, "corpora/environmentalhealthai-56dtvvug606r/documents/environmenthealthaihackatho-upwnfs1zk9ew")

    #   Setting up the project itself
    model = UserQuery()

    response = asyncio.run(model.user_query(user_query, explanation_description, relevant_chunk_count))

    print("\n\n\n", response, "\n\n\n")