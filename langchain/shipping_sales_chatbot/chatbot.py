import gradio as gr

from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

def initialize_faiss():
    with open("data.txt", "r", encoding="UTF-8") as f:
        real_estate_sales = f.read()
    text_splitter = CharacterTextSplitter(        
        separator = r'\d+\.',
        chunk_size = 150,
        chunk_overlap = 10,
        length_function = len,
        is_separator_regex = True,
    )
    docs = text_splitter.create_documents([real_estate_sales])
    db = FAISS.from_documents(docs, OpenAIEmbeddings())
    db.save_local("shipping_sale")

def initialize_sales_bot(vector_store_dir: str="shipping_sale"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    global LLM
    LLM = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(LLM,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        if len(ans['source_documents']) == 0:
            conversation = [("system", "你是一个在航运公司工作了十多年的销售，拥有丰富的航运知识。你需要回复用户咨询的关于航运的问题。")]
            # 将history作为上下文
            for user_msg, ai_response in history:
                conversation.append(("user", user_msg))
                conversation.append(("ai", ai_response))
            conversation.append(("user", message))
            print(f"[conversation]{conversation}")
            prompt = ChatPromptTemplate.from_messages(conversation)
            chat = prompt.format_messages()
            chat_result = LLM.invoke(chat)
            print(f"[chat_result.content]{chat_result.content}")
            return chat_result.content
        else:
            return ans["result"]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="航运销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=False, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化数据
    initialize_faiss()
    # 初始化航运销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
