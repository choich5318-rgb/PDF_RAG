# Ollama_PDF_RAG.py - 다중 PDF 기반 RAG 시스템
import gradio as gr
import ollama
import os
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
try:
    from langchain_core.documents import Document
except ImportError:
    # Fallback for older langchain versions
    from langchain.schema import Document

# 설정
VECTOR_CACHE_DIR = "chroma_pdf_cache"
CACHE_METADATA_FILE = os.path.join(VECTOR_CACHE_DIR, "cache_metadata.json")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"
DEFAULT_LLM_MODEL = "llama3"

# 캐시 메타데이터 관리
def load_cache_metadata() -> Dict:
    """캐시 메타데이터 로드"""
    if os.path.exists(CACHE_METADATA_FILE):
        try:
            with open(CACHE_METADATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache_metadata(metadata: Dict):
    """캐시 메타데이터 저장"""
    os.makedirs(VECTOR_CACHE_DIR, exist_ok=True)
    with open(CACHE_METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def get_file_hash(file_path: str) -> str:
    """파일의 MD5 해시 계산"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_collection_name(file_path: str) -> str:
    """파일 경로에서 안전한 컬렉션 이름 생성"""
    file_name = Path(file_path).stem
    # 특수 문자 제거 및 길이 제한
    safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in file_name)
    return safe_name[:50]  # Chroma 컬렉션 이름 길이 제한

# PDF 문서 로드 및 벡터화
def load_and_vectorize_pdf(
    file_path: str, 
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    progress_callback=None
) -> Tuple[Chroma, str]:
    """
    PDF를 로드하고 벡터화합니다.
    이미 처리된 파일은 캐시에서 로드합니다.
    
    Returns:
        (vectorstore, collection_name)
    """
    file_hash = get_file_hash(file_path)
    collection_name = get_collection_name(file_path)
    cache_metadata = load_cache_metadata()
    
    # 캐시 확인
    if file_path in cache_metadata:
        cached_info = cache_metadata[file_path]
        if cached_info.get('hash') == file_hash:
            # 캐시된 벡터 스토어 로드
            if progress_callback:
                progress_callback(f"📂 캐시에서 로드 중: {Path(file_path).name}")
            
            embeddings = OllamaEmbeddings(model=embedding_model)
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=VECTOR_CACHE_DIR
            )
            return vectorstore, collection_name
    
    # 새로 처리
    if progress_callback:
        progress_callback(f"📄 PDF 로드 중: {Path(file_path).name}")
    
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    if not docs:
        raise ValueError(f"❗ PDF에서 텍스트를 추출할 수 없습니다: {Path(file_path).name}")

    if progress_callback:
        progress_callback(f"✂️ 텍스트 분할 중... (총 {len(docs)} 페이지)")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)
    
    # 메타데이터에 파일 경로 추가
    for split in splits:
        split.metadata['source_file'] = Path(file_path).name
    
    if progress_callback:
        progress_callback(f"🔢 임베딩 생성 중... (총 {len(splits)} 청크)")

    embeddings = OllamaEmbeddings(model=embedding_model)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=VECTOR_CACHE_DIR
    )
    vectorstore.persist()

    # 캐시 메타데이터 업데이트
    cache_metadata[file_path] = {
        'hash': file_hash,
        'collection_name': collection_name,
        'chunks': len(splits),
        'pages': len(docs)
    }
    save_cache_metadata(cache_metadata)

    if progress_callback:
        progress_callback(f"✅ 완료: {Path(file_path).name} ({len(splits)} 청크)")

    return vectorstore, collection_name

# 여러 PDF에서 통합 검색
def search_multiple_pdfs(
    file_paths: List[str],
    question: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = 5,
    progress_callback=None
) -> List[Document]:
    """
    여러 PDF에서 질문에 대한 관련 문서를 검색합니다.
    """
    all_results = []
    
    for file_path in file_paths:
        try:
            vectorstore, _ = load_and_vectorize_pdf(file_path, embedding_model, progress_callback)
            retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
            docs = retriever.invoke(question)
            
            # 각 문서에 출처 정보 추가
            for doc in docs:
                doc.metadata['source_file'] = Path(file_path).name
                all_results.append(doc)
                
        except Exception as e:
            if progress_callback:
                progress_callback(f"⚠️ 오류 ({Path(file_path).name}): {str(e)}")
            continue
    
    # 관련도 순으로 정렬 (여러 PDF 결과 통합)
    # 중복 제거 (유사한 내용)
    unique_results = []
    seen_content = set()
    
    for doc in all_results:
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_results.append(doc)
    
    # 상위 결과만 반환
    return unique_results[:top_k * len(file_paths)]

# 문서 포맷팅 (출처 포함)
def format_docs_with_source(docs: List[Document]) -> str:
    """문서를 포맷팅하고 출처 정보를 포함합니다."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source_file', '알 수 없음')
        page = doc.metadata.get('page', 'N/A')
        content = doc.page_content.strip()
        formatted.append(f"[출처 {i}: {source} (페이지 {page})]\n{content}")
    return "\n\n---\n\n".join(formatted)

# RAG 체인 동작 (다중 PDF 지원)
def rag_chain_multiple(
    files: List,
    question: str,
    llm_model: str = DEFAULT_LLM_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    progress=gr.Progress()
) -> str:
    """
    여러 PDF 파일에서 질문에 대한 답변을 생성합니다.
    """
    if not files:
        return "❌ PDF 파일을 업로드해주세요."
    
    if not question or not question.strip():
        return "❌ 질문을 입력해주세요."
    
    try:
        file_paths = [f.name for f in files if f]
        
        if not file_paths:
            return "❌ 유효한 PDF 파일이 없습니다."
        
        progress(0, desc=f"📚 {len(file_paths)}개 PDF 처리 중...")
        
        # 여러 PDF에서 검색
        def update_progress(msg):
            progress(0.5, desc=msg)
        
        retrieved_docs = search_multiple_pdfs(
            file_paths, 
            question, 
            embedding_model,
            top_k=5,
            progress_callback=update_progress
        )

        if not retrieved_docs:
            return "❌ 관련 문서를 찾을 수 없습니다. 질문을 더 구체적으로 작성해 보거나 다른 PDF를 사용해 보세요."

        progress(0.8, desc="🤖 LLM 답변 생성 중...")
        
        context = format_docs_with_source(retrieved_docs)
        
        # 사용된 PDF 목록
        used_files = list(set([doc.metadata.get('source_file', '알 수 없음') for doc in retrieved_docs]))
        
        prompt = f"""다음은 여러 PDF 문서에서 검색된 관련 내용입니다:

{context}

질문: {question}

위의 문서 내용을 바탕으로 질문에 답변해주세요. 답변은 한국어로 작성하고, 이모지를 사용하여 가독성을 높여주세요. 
답변의 마지막에 참고한 문서 출처를 명시해주세요."""

        response = ollama.chat(
            model=llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on PDF documents. Always answer in Korean and use emojis to make the response more readable."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        answer = response['message']['content']
        
        # 출처 정보 추가
        answer += f"\n\n📚 참고 문서: {', '.join(used_files)}"
        
        progress(1.0, desc="✅ 완료!")
        
        return answer

    except Exception as e:
        error_msg = str(e)
        return f"❌ 오류 발생: {error_msg}\n\n오류가 계속되면 다른 PDF 파일을 시도해보세요."

# 사용 가능한 모델 목록 가져오기
def get_available_models():
    """Ollama에서 사용 가능한 모델 목록 가져오기"""
    try:
        models = ollama.list()
        llm_models = [m['name'] for m in models.get('models', [])]
        return llm_models if llm_models else [DEFAULT_LLM_MODEL]
    except:
        return [DEFAULT_LLM_MODEL]

# Gradio 인터페이스
def create_interface():
    """Gradio 인터페이스 생성"""
    available_models = get_available_models()
    
    with gr.Blocks(title="다중 PDF RAG 시스템") as iface:
        gr.Markdown("# 📚 다중 PDF 기반 질문 응답 시스템")
        gr.Markdown("여러 PDF 파일을 업로드하고 질문을 입력하면, 모든 PDF에서 관련 내용을 검색하여 답변해드립니다.")
        
        with gr.Row():
            with gr.Column(scale=2):
                files = gr.File(
                    label="PDF 파일 업로드 (여러 개 선택 가능)",
                    file_count="multiple",
                    file_types=[".pdf"],
                    type="filepath"
                )
                
                question = gr.Textbox(
                    label="질문을 입력하세요",
                    placeholder="예: 이 문서들의 주요 내용은 무엇인가요?",
                    lines=3
                )
                
                with gr.Row():
                    llm_model = gr.Dropdown(
                        label="LLM 모델",
                        choices=available_models,
                        value=DEFAULT_LLM_MODEL if DEFAULT_LLM_MODEL in available_models else available_models[0],
                        interactive=True
                    )
                    
                    embedding_model = gr.Dropdown(
                        label="임베딩 모델",
                        choices=["mxbai-embed-large", "nomic-embed-text"],
                        value=DEFAULT_EMBEDDING_MODEL,
                        interactive=True
                    )
                
                submit_btn = gr.Button("🔍 질문하기", variant="primary", size="lg")
                
            with gr.Column(scale=3):
                output = gr.Textbox(
                    label="답변",
                    lines=15,
                    max_lines=20
                )
                
                with gr.Accordion("📋 처리된 파일 목록", open=False):
                    file_list = gr.Textbox(
                        label="",
                        lines=5,
                        interactive=False
                    )
        
        # 캐시 관리
        with gr.Accordion("⚙️ 캐시 관리", open=False):
            with gr.Row():
                clear_cache_btn = gr.Button("🗑️ 캐시 초기화", variant="stop")
                cache_info = gr.Textbox(label="캐시 정보", interactive=False)
        
        def clear_cache():
            """캐시 초기화"""
            try:
                import shutil
                if os.path.exists(VECTOR_CACHE_DIR):
                    shutil.rmtree(VECTOR_CACHE_DIR)
                os.makedirs(VECTOR_CACHE_DIR, exist_ok=True)
                return "✅ 캐시가 초기화되었습니다."
            except Exception as e:
                return f"❌ 오류: {str(e)}"
        
        def update_file_list(files):
            """파일 목록 업데이트"""
            if not files:
                return ""
            file_names = [Path(f.name).name for f in files if f]
            return "\n".join([f"• {name}" for name in file_names])
        
        def get_cache_info():
            """캐시 정보 가져오기"""
            metadata = load_cache_metadata()
            if not metadata:
                return "캐시된 파일이 없습니다."
            
            info_lines = [f"캐시된 파일: {len(metadata)}개"]
            for file_path, info in metadata.items():
                name = Path(file_path).name
                chunks = info.get('chunks', 0)
                pages = info.get('pages', 0)
                info_lines.append(f"  • {name}: {pages}페이지, {chunks}청크")
            
            return "\n".join(info_lines)
        
        # 이벤트 핸들러
        submit_btn.click(
            fn=rag_chain_multiple,
            inputs=[files, question, llm_model, embedding_model],
            outputs=output
        )
        
        files.change(
            fn=update_file_list,
            inputs=files,
            outputs=file_list
        )
        
        clear_cache_btn.click(
            fn=clear_cache,
            outputs=cache_info
        )
        
        iface.load(
            fn=get_cache_info,
            outputs=cache_info
        )
    
    return iface

if __name__ == "__main__":
    # 캐시 디렉토리 생성
    os.makedirs(VECTOR_CACHE_DIR, exist_ok=True)
    
    iface = create_interface()
    # server_name을 localhost로 설정 (로컬 접근용)
    # 원격 접근이 필요한 경우 "0.0.0.0"으로 변경하고 http://localhost:7860 또는 실제 IP로 접근
    iface.launch(share=False, server_name="127.0.0.1", server_port=7860)
