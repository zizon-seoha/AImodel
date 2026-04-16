# 필요한 라이브러리 임포트
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. 문서 로드
loader = PyPDFLoader(r"C:\Users\master\Desktop\GSM_길잡이_RAG_지식베이스클로드.pdf")
docs = loader.load()

# 2. 텍스트 분할 (Chunk size: 1000자, Overlap: 100자)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
splits = text_splitter.split_documents(docs)

# 3. 임베딩 모델 정의 (OpenAI 사용 예시)
model_name = "jhgan/ko-sroberta-multitask"

# 임베딩 객체 생성
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'}, # GPU가 있다면 'cuda'로 변경하세요
    encode_kwargs={'normalize_embeddings': True} # 코사인 유사도 검색을 위해 True 설정 권장
)

# 4. 임베딩 후 벡터 DB(FAISS)에 저장
vectorstore = FAISS.from_documents(splits, embeddings)

# (선택) 구축된 인덱스를 로컬에 저장하여 나중에 재사용 가능
vectorstore.save_local("faiss_index")
# 사용자 질문 예시 (문서 내용에 맞는 질문으로 바꿔보세요)
query = "학교 생활 꿀팁"

# 벡터 DB에서 질문과 가장 유사한 텍스트 덩어리 3개(k=3) 찾아오기
retrieved_docs = vectorstore.similarity_search(query, k=3)

# 찾아온 내용 출력해보기
print("🔍 검색 결과 확인:")
for i, doc in enumerate(retrieved_docs):
    print(f"\n--- [관련 문서 {i+1} (페이지: {doc.metadata.get('page', '알수없음')})] ---")
    print(doc.page_content)