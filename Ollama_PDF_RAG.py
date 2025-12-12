# Ollama_PDF_RAG.py
import gradio as gr
import ollama
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# 캐시 디렉토리
VECTOR_CACHE_DIR = "chroma_pdf_cache"

# PDF 문서 로드 및 벡터화
def load_and_retrieve_pdf(file_path: str):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    if not docs:
        raise ValueError("❗ PDF에서 텍스트를 추출할 수 없습니다. 다른 파일을 시도해 보세요.")

    print(f"PDF 문서 로드 완료. 첫 페이지 미리보기:\n{docs[0].page_content[:300]}...\n")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=VECTOR_CACHE_DIR
    )
    vectorstore.persist()

    return vectorstore.as_retriever()

# 문서 포맷팅
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG 체인 동작
def rag_chain(file, question: str) -> str:
    try:
        retriever = load_and_retrieve_pdf(file.name)
        retrieved_docs = retriever.invoke(question)

        if not retrieved_docs:
            return "관련 문서를 찾을 수 없습니다. 질문을 더 구체적으로 작성해 보거나 다른 PDF를 사용해 보세요."

        context = format_docs(retrieved_docs)
        print(f"검색된 문맥 미리보기:\n{context[:500]}...\n")

        prompt = f"Question: {question}\n\nContext: {context}"

        response = ollama.chat(
            model='llama3',
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Read the PDF content and answer the question. Translate the answer in Korean with emoji."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response['message']['content']

    except Exception as e:
        return f"❌ 오류 발생: {str(e)}"

# Gradio 인터페이스
iface = gr.Interface(
    fn=rag_chain,
    inputs=[
        gr.File(label="PDF 파일 업로드", type="filepath"),
        gr.Textbox(label="질문을 입력하세요")
    ],
    outputs="text",
    title="LLaMA 3 - PDF 기반 질문 응답 (개선 버전)",
    description="PDF 파일을 올리고 질문을 입력하면, 해당 내용을 기반으로 LLaMA 3가 한국어로 답변해 줍니다."
)

if __name__ == "__main__":
    iface.launch()
