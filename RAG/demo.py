from vectorbase import VectorStore
from utils import ReadFiles
from llm_model import OpenAIChat
from embeddings import OpenAIEmbedding


docs = ReadFiles('/home/zhanghudong/llm/RAG/data').get_content(max_token_len=600, cover_content=150)
vector = VectorStore(docs)
embedding = OpenAIEmbedding() 
vector.get_vector(EmbeddingModel=embedding)
vector.persist(path='storage')


question = 'what is the data augmentation?'

content = vector.query(question, EmbeddingModel=embedding, k=1)
chat = OpenAIChat(model='Qwen/Qwen2.5-32B-Instruct')
print(chat.chat(question, [], content))
