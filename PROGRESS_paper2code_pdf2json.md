# 진행 기록 — pdf2json skill 임베딩 + paper2code 실행

최종 업데이트: 2026-06-05
작업 폴더: `C:\Users\rudtn\Desktop\Between Laptop\Github`

다음 세션이 이 파일을 먼저 읽으면 맥락 복원 + 이어서 진행 가능.

---

## 1. 한 일 요약 (2건)

1. **`pdf2json` skill 신규 제작** — PDF/arxiv → S2ORC JSON 변환 (allenai s2orc-doc2json + 로컬 Grobid docker).
2. **`paper2code`에 s2orc 백엔드 통합** — opt-in 환경변수로 고품질 파서 선택 가능.
3. **`/paper2code 1803.08475` 실행 완료** — "Attention, Learn to Solve Routing Problems!" (Kool et al., ICLR 2019) TSP 구현 생성. 결과물 = `attention_learn_to_route/`.

---

## 2. pdf2json skill

### 위치
- skill: `.claude/skills/pdf2json/` (`SKILL.md` + `scripts/pdf2json.py`)
  - 주의: 다른 skill은 `.agents/skills/<name>` 실체 + `.claude/skills/<name>` 심볼릭 링크.
    pdf2json은 `.claude/skills/`에 실폴더로 둠 (Claude Code가 거기도 탐색하므로 동작).
- 변환 엔진(원본): `C:\Users\rudtn\s2orc-doc2json` (allenai 패키지).

### 동작 방식
- `python .claude/skills/pdf2json/scripts/pdf2json.py -i <PDF|arxiv id|URL> -o <outdir>`
- arxiv면 PDF 자동 다운로드 → Grobid 컨테이너 자동 기동(`grobid_pdf2json`, port 8070) →
  PDF→TEI.XML→S2ORC JSON. 마지막 stdout 줄 = 결과 JSON 경로.

### 환경 셋업 (이미 완료된 것)
- `doc2json` editable 설치됨: `pip install -e C:\Users\rudtn\s2orc-doc2json --no-deps` →
  어디서든 `import doc2json` 가능. (deps는 Python 3.14 환경에 이미 충족)
- Grobid 이미지 pull 완료: `lfoppiano/grobid:0.7.3` (docker 29.3.1, Java 17 확인됨).
- **비용 0원**. Grobid = 오픈소스 자체호스팅, RAM ~2-4GB만 점유. API료/네트워크 업로드 없음.

### 발견·수정한 함정 (재발 주의)
1. **Windows `split('/')` 버그**: upstream `process_pdf_file`이 paper_id를 POSIX 전용
   `input_file.split('/')`로 추출 → 역슬래시 경로서 빈 `AssertionError`.
   → 래퍼/통합 코드가 경로를 forward-slash로 정규화해 우회.
2. **Grobid cold-start**: `/api/isalive`가 fulltext 모델 로드 *전에* true 반환 →
   첫 파싱 실패 가능. 30-60초 후 재시도하면 warm 컨테이너 재사용해 성공.
3. **cp949 콘솔**: ✓ 같은 유니코드 출력시 `UnicodeEncodeError`. `python -X utf8`로 실행.
   (파일 IO엔 무관, 표시만. CLAUDE.md에도 명시됨)

### 컨테이너 관리
- `--rm`, 이름 `grobid_pdf2json`, port 8070. 런 간 재사용됨.
- 중지: `docker stop grobid_pdf2json` (현재 중지 상태로 둠).

---

## 3. paper2code s2orc 백엔드 통합

### 수정 파일
- `.agents/skills/paper2code/scripts/fetch_paper.py`:
  `extract_with_s2orc()` 추가 + main()에서 1순위 시도(opt-in).
- `.agents/skills/paper2code/SKILL.md`: Stage 1에 사용법 문서화.

### 사용법
- `PAPER2CODE_PARSER=s2orc` 환경변수 설정 후 `fetch_paper.py` 실행 → s2orc 파서 사용.
- Docker/Grobid 없으면 **자동 무음 fallback** → pymupdf4llm(기본).
- env override: `PDF2JSON_SCRIPT`, `S2ORC_HOME`, `GROBID_PORT`.

### 트레이드오프 (중요 — 백엔드 선택 기준)
| 백엔드 | 섹션구조 | 수식 LaTeX | 코드링크 탐지 |
|--------|---------|-----------|------------|
| **s2orc** (opt-in) | 우수(41섹션) | **손실(평탄화)** | ✓ |
| **pymupdf4llm** (기본) | 평탄 | **보존** | 텍스트스캔만 |
- Grobid은 인라인 수식 LaTeX(`\frac`, 아래첨자) 평탄화 → **수식 많은 논문엔 pymupdf4llm 기본 유지**.
- 섹션/참고문헌 품질만 필요한 수식-적은 논문엔 s2orc 권장.

---

## 4. 생성된 구현 — `attention_learn_to_route/`

### 대상 논문
"Attention, Learn to Solve Routing Problems!" — Kool, van Hoof, Welling (ICLR 2019).
arxiv 1803.08475. 공식코드: github.com/wouterkool/attention-learn-to-route (PyTorch).

### 구조
```
attention_learn_to_route/
├── configs/base.yaml          # 하이퍼파라미터 (전부 §-인용)
├── src/
│   ├── model.py               # §3 Attention Model: Encoder + Decoder
│   ├── baseline.py            # §4 greedy rollout baseline + t-test 갱신
│   ├── loss.py                # §4 REINFORCE loss
│   ├── train.py               # Algorithm 1 학습 루프
│   ├── evaluate.py            # §5 greedy/sampling 디코딩, optimality gap
│   ├── data.py                # §5/App.B.2 TSP 인스턴스 생성
│   └── utils.py               # config 로딩, tour length L(π)
├── notebooks/walkthrough.ipynb # 논문↔코드 18셀, 실행가능 sanity check
├── README.md
├── REPRODUCTION_NOTES.md       # 명시/미명시 정직 매핑
└── requirements.txt
```

### 범위
- **TSP만** 구현 (논문이 TSP로 모델 정의). VRP/OP/PCTSP는 동일 모델·다른 mask/context → scope 밖.
- minimal 모드 | pytorch.

### 핵심 스펙 (논문 인용 기반)
- d_h=128, N=3 layers, M=8 heads, d_k=d_v=16, d_ff=512, BatchNorm(LayerNorm 아님), tanh clip C=10.
- 학습: Adam lr=1e-4 상수, 100ep × 2500step × 512 batch, init U(-1/√d,1/√d).
- baseline: greedy rollout(frozen best model), per-epoch 일방 paired t-test(α=5%, 10000 instances),
  1epoch만 exponential warmup β=0.8.

### 정직히 처리한 함정 (REPRODUCTION_NOTES.md 표 참조)
- `d_k = d_h/M` (App.A의 "Md_h=16"은 OCR오류, 128/8=16만 일관).
- tanh clip은 mask **전**에 적용.
- advantage `L(π)-b(s)` detach (상수 가중치).
- 미명시 6개: Adam betas/eps, weight_decay, seed, QKV bias, grad_clip(=1.0 from official code).

### 검증 (실제 실행 완료)
- 스모크: forward/loss/backward, tour 유효순열, grad 흐름 ✓
- 통합: 5스텝 비용 10→5.6, t-test 갱신 동작, sampling<greedy ✓
- 노트북: 전 18셀 실행, 모든 assert 통과, held-out greedy 5.21→3.61 (학습 확인) ✓

---

## 5. 환경 상태 (이번 세션서 설치됨)
- `torch==2.12.0+cpu`, `scipy==1.17.1`, `pyyaml` 설치됨.
- `doc2json` editable 설치됨.
- Grobid docker 이미지 있음(중지 상태).
- Python 3.14.3.

---

## 6. 다음에 할 수 있는 것 (미완/확장)
- [ ] `attention_learn_to_route` 실제 학습 돌려 TSP-20 결과 재현 (GPU 권장; 논문 ~3.84 길이 근접 목표).
      CPU면 config에서 `n_epochs`/`steps_per_epoch` 축소.
- [ ] VRP/OP/PCTSP 변형 추가 (mask/context/objective만 교체 — 동일 인코더·디코더 재사용).
- [ ] pdf2json: Grobid `consolidate_header` 켜면 title/authors 메타 채워짐 (CrossRef 네트워크 필요).
- [ ] pdf2json skill을 `.agents/skills/`로 옮기고 `.claude`에 심볼릭 링크 (다른 skill 컨벤션 일치) — 선택.
- [ ] paper2code 결과물 git repo로 만들어 GitHub(kyoungsoon-kim) 푸시 — 사용자 지시 시.

---

## 7. 메모리
- `~/.claude/projects/.../memory/pdf2json-skill.md`에 핵심 사실 저장됨 (MEMORY.md 인덱스 등록).
