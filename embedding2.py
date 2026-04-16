import os
import csv
import json
import pickle
from pathlib import Path
from typing import Optional
 
import torch
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
 

CSV_PATH = r"file:///C:/Users/master/Documents/rag_document%20(1).pdf"
EMBED_MODEL     = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"   # 한국어 SBERT
GEN_MODEL       = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"  # 한국어 LLM
FAISS_INDEX     = "gsm_faiss.index"
CHUNKS_PATH     = "gsm_chunks.pkl"
TOP_K           = 5      # 검색할 청크 수
MAX_NEW_TOKENS  = 512    # 생성 최대 토큰
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
 

NOISE_ANSWERS = {
    "엄.. 기억이 안나요", "딱히 없습니다", "딱히 없었던거 같은데",
    "흠.. 한번에 성공해서...", "없음", "-", "n/a", "",
    "놀지 않고 공부만 하기",
}
 
# 제외할 컬럼
EXCLUDE_COLS = {
    "타임스탬프", "이메일 주소",
    "더 하고 싶은 말이 있으시다면 카테고리를 클릭하고 \n작성해주십쇼!",
    "기능반이 하는 일 또는 기능반의 목적\n(기능반만 작성 해주시면 됩니다)",
}
EXCLUDE_PARTIAL = [
    "기능반을 하면 감수해야",
    "로보틱스가 무엇",
    "사이버보안이 무엇",
    "모바일 앱 개발이 무엇",
    "클라우드 컴퓨팅",
]
 
# 컬럼명 → 사람이 읽기 좋은 카테고리명
COL_DISPLAY = {
    "후배들에게 가장 해주고 싶은 조언\n최대한 구체적으로 작성해주시면 큰 도움이 됩니다!\n(예: 공부 방법, 마음가짐, 실수 경험 등)":
        "후배들에게 가장 해주고 싶은 조언",
    "학교에 들어오고 나서 가장 먼저 하면 좋은 것":
        "학교에 들어오고 나서 가장 먼저 하면 좋은 것",
    "가장 힘들었던 경험을 해결했던 방법":
        "가장 힘들었던 경험을 해결했던 방법",
    "프로젝트 경험과 프로젝트를 하면서 힘들었던 점":
        "프로젝트 경험과 힘들었던 점",
    "공기업을 가기 위해 해야할 것(필수 X)":
        "공기업 취업을 위해 해야 할 것",
    "학교 생활 꿀팁(필수 X)":
        "학교 생활 꿀팁",
    "IT 네트워크가 무엇인지 또는 주로 하는 일":
        "IT 네트워크 기능반 소개",
    "공부에 대한 조언":   "공부에 대한 조언",
    "프로젝트에 대한 조언": "프로젝트에 대한 조언",
    "인간관계에 대한 조언": "인간관계에 대한 조언",
    "진로에 대한 조언":   "진로에 대한 조언",
    "기타 조언":          "기타 조언",
}
 
 
# ──────────────────────────────────────────────
# 1. 데이터 로딩 & 청킹
# ──────────────────────────────────────────────
def load_chunks(csv_path: str) -> list[dict]:
    """
    CSV → 청크 리스트 변환.
    청크 단위: (질문 카테고리, 응답자 정보, 답변 텍스트)
    → RAG 검색 단위를 '한 응답자의 한 질문 답변'으로 설정.
    """
    chunks = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grade  = row.get("기수 / 학년", "").strip()
            status = row.get("현재 상태", "").strip()
            major  = row.get("전공", "").strip()
            meta   = f"[{grade} / {status} / {major}]"
 
            for col, val in row.items():
                # 제외 컬럼
                if col in EXCLUDE_COLS:
                    continue
                if col in {"기수 / 학년", "현재 상태", "전공"}:
                    continue
                if any(col.startswith(p) for p in EXCLUDE_PARTIAL):
                    continue
 
                val = val.strip()
                if not val or val.lower() in NOISE_ANSWERS or len(val) < 8:
                    continue
 
                category = COL_DISPLAY.get(col, col.replace("\n", " ").strip())
 
                # 청크 텍스트: 검색과 생성 모두에 활용되는 핵심 텍스트
                chunk_text = (
                    f"카테고리: {category}\n"
                    f"응답자 정보: {meta}\n"
                    f"답변: {val}"
                )
 
                chunks.append({
                    "text":     chunk_text,       # 임베딩 대상
                    "category": category,
                    "grade":    grade,
                    "status":   status,
                    "major":    major,
                    "answer":   val,              # 원본 답변 (컨텍스트용)
                })
 
    print(f"✅ 총 {len(chunks)}개 청크 생성 완료")
    return chunks
 
 
# ──────────────────────────────────────────────
# 2. FAISS 인덱스 빌드 / 로드
# ──────────────────────────────────────────────
def build_index(chunks: list[dict], embed_model: SentenceTransformer) -> faiss.Index:
    print("🔄 청크 임베딩 중...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # 코사인 유사도를 내적으로 계산
    )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product (= 코사인, 정규화됨)
    index.add(embeddings.astype(np.float32))
    print(f"✅ FAISS 인덱스 빌드 완료 (dim={dim}, 벡터 수={index.ntotal})")
    return index
 
 
def save_index(index: faiss.Index, chunks: list[dict]):
    faiss.write_index(index, FAISS_INDEX)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"💾 인덱스 저장 완료: {FAISS_INDEX}, {CHUNKS_PATH}")
 
 
def load_index() -> tuple[faiss.Index, list[dict]]:
    index  = faiss.read_index(FAISS_INDEX)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    print(f"📂 인덱스 로드 완료 (벡터 수={index.ntotal})")
    return index, chunks
 
 
# ──────────────────────────────────────────────
# 3. 검색
# ──────────────────────────────────────────────
def retrieve(
    query: str,
    embed_model: SentenceTransformer,
    index: faiss.Index,
    chunks: list[dict],
    top_k: int = TOP_K,
) -> list[dict]:
    q_vec = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
 
    scores, indices = index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({**chunks[idx], "score": float(score)})
    return results
 
 
# ──────────────────────────────────────────────
# 4. 프롬프트 조합
# ──────────────────────────────────────────────
def build_prompt(query: str, retrieved: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(retrieved, 1):
        context_parts.append(
            f"[참고 {i}] 카테고리: {chunk['category']} | "
            f"{chunk['grade']} / {chunk['status']} / {chunk['major']}\n"
            f"{chunk['answer']}"
        )
    context = "\n\n".join(context_parts)
 
    prompt = f"""당신은 광주소프트웨어마이스터고등학교(GSM)의 선배들이 작성한 조언과 경험을 바탕으로 후배 학생들의 질문에 답변하는 AI 길잡이입니다.
 
아래 [참고 자료]는 실제 선배들이 작성한 답변입니다. 이를 종합하여 질문에 친근하고 도움이 되는 답변을 작성해주세요.
 
규칙:
- 선배들의 실제 경험과 조언을 바탕으로 답변하세요.
- 없는 내용을 지어내지 마세요.
- 참고 자료에 없는 내용이라면 솔직하게 모른다고 하세요.
- 친근한 말투로 답변하세요.
- 답변은 300자 내외로 핵심만 정리하세요.
 
[참고 자료]
{context}
 
[질문]
{query}
 
[답변]"""
    return prompt
 
 
# ──────────────────────────────────────────────
# 5. 생성
# ──────────────────────────────────────────────
def generate_answer(
    prompt: str,
    gen_pipeline,
) -> str:
    outputs = gen_pipeline(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        return_full_text=False,
    )
    return outputs[0]["generated_text"].strip()
 
 
# ──────────────────────────────────────────────
# 6. 메인 파이프라인
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  GSM 길잡이 AI 챗봇 - RAG 파이프라인")
    print("=" * 60)
 
    # ── 임베딩 모델 로드 ──
    print(f"\n📦 임베딩 모델 로드 중: {EMBED_MODEL}")
    embed_model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
 
    # ── FAISS 인덱스 빌드 or 로드 ──
    if Path(FAISS_INDEX).exists() and Path(CHUNKS_PATH).exists():
        print("\n📂 기존 인덱스 발견 → 로드합니다.")
        index, chunks = load_index()
    else:
        print(f"\n📄 CSV 로드 중: {CSV_PATH}")
        chunks = load_chunks(CSV_PATH)
        index  = build_index(chunks, embed_model)
        save_index(index, chunks)
 
    # ── 생성 모델 로드 ──
    print(f"\n📦 생성 모델 로드 중: {GEN_MODEL}")
    print("   (처음 실행 시 수 분 소요될 수 있습니다)")
 
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    print("✅ 모든 모델 로드 완료!\n")
 
    # ── 챗봇 루프 ──
    print("💬 GSM 길잡이 챗봇을 시작합니다.")
    print("   종료하려면 'quit' 또는 'q'를 입력하세요.\n")
    print("-" * 60)
 
    while True:
        try:
            query = input("\n🙋 질문: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n챗봇을 종료합니다.")
            break
 
        if not query:
            continue
        if query.lower() in {"quit", "q", "종료", "exit"}:
            print("챗봇을 종료합니다. 화이팅! 💪")
            break
 
        # 검색
        retrieved = retrieve(query, embed_model, index, chunks)
 
        print(f"\n📎 관련 청크 {len(retrieved)}개 검색됨:")
        for i, r in enumerate(retrieved, 1):
            print(f"   {i}. [{r['category']}] {r['grade']} / {r['status']} / {r['major']} (유사도: {r['score']:.3f})")
 
        # 생성
        prompt = build_prompt(query, retrieved)
        print("\n🤖 답변 생성 중...")
        answer = generate_answer(prompt, gen_pipe)
 
        print(f"\n{'─'*60}")
        print(f"🎓 길잡이 답변:\n{answer}")
        print(f"{'─'*60}")
 
 
# ──────────────────────────────────────────────
# 검색만 테스트 (생성 모델 없이)
# ──────────────────────────────────────────────
def test_retrieval_only():
    """
    생성 모델 없이 검색 결과만 확인하고 싶을 때 사용.
    실행: python gsm_chatbot_rag.py --retrieval-only
    """
    print("=" * 60)
    print("  GSM RAG - 검색 테스트 모드 (생성 모델 없음)")
    print("=" * 60)
 
    embed_model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
 
    if Path(FAISS_INDEX).exists() and Path(CHUNKS_PATH).exists():
        index, chunks = load_index()
    else:
        chunks = load_chunks(CSV_PATH)
        index  = build_index(chunks, embed_model)
        save_index(index, chunks)
 
    test_queries = [
        "학교 처음 와서 뭘 해야 해?",
        "슬럼프 어떻게 극복해?",
        "프로젝트 어렵게 느껴지는데 어떻게 해?",
        "공기업 가려면 뭘 준비해야 해?",
        "선배들이랑 어떻게 친해져?",
        "AI 써서 개발해도 돼?",
    ]
 
    for q in test_queries:
        print(f"\n❓ 질문: {q}")
        results = retrieve(q, embed_model, index, chunks)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] (유사도:{r['score']:.3f}) [{r['category']}] {r['grade']}/{r['major']}")
            print(f"       {r['answer'][:80]}...")
        print()
 
 
# ──────────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if "--retrieval-only" in sys.argv:
        test_retrieval_only()
    else:
        main()