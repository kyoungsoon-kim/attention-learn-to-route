# Attention, Learn to Solve Routing Problems! — TSP & CVRP 재구현

Kool, van Hoof, Welling (ICLR 2019)의 **Attention Model(AM)** 과 **REINFORCE + greedy
rollout baseline** 학습을 **유클리드 TSP** 와 **CVRP** 두 문제에 대해 충실히(citation-anchored)
재구현한 PyTorch 프로젝트입니다. 하나의 인코더·디코더 코어를 두 문제가 공유하도록 설계했습니다.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.6%2Bcu124-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Paper" src="https://img.shields.io/badge/arXiv-1803.08475-b31b1b">
  <img alt="Problems" src="https://img.shields.io/badge/problems-TSP%20%7C%20CVRP-2C7BE5">
  <img alt="RL" src="https://img.shields.io/badge/method-REINFORCE%20%2B%20rollout%20baseline-6f42c1">
</p>

> Kool, W., van Hoof, H., & Welling, M. (2019). *Attention, Learn to Solve Routing
> Problems!* ICLR 2019. [arXiv:1803.08475](https://arxiv.org/abs/1803.08475)

---

## 📌 왜 이 문제인가

TSP·CVRP 같은 라우팅 문제는 NP-hard 라 정확해(Concorde·LKH)는 규모가 커지면 비싸지고, 사람이
만든 휴리스틱은 문제마다 새로 설계해야 합니다. 이 논문은 **그래프를 attention 으로 인코딩하고,
방문 순서를 autoregressive 하게 생성하는 정책을 강화학습으로 학습**해, 한 번 학습하면 새 인스턴스를
한 번의 forward pass로 빠르게 푸는 방법을 제안합니다.

이 저장소는 그 방법을 **검증 가능한 최소 구현**으로 옮긴 것입니다. 논문의 모든 클래스·비자명
함수에 `§`-섹션 인용을 달았고, 논문이 명시하지 않은 선택은 `[UNSPECIFIED]` /
`[FROM_OFFICIAL_CODE]` 로 정직하게 표기했습니다(상세 [`REPRODUCTION_NOTES.md`](REPRODUCTION_NOTES.md)).

## 🧠 동작 원리

전체 파이프라인은 **인코더 → 디코더(autoregressive rollout) → REINFORCE** 세 단계입니다.

```
 입력 인스턴스                Encoder (§3.1)                   Decoder (§3.2, 매 스텝 반복)
 (좌표 / +demand)   ─►  embedder + N×[ MHA + FF, BN ]  ─►   context ─► M-head glimpse
                                    │                              │
                              node_emb (B,n,d_h)            single-head pointer
                              graph_emb (B,d_h)             (tanh clip C=10 → mask)
                                                                   │
                                                            softmax → 노드 선택
                                                                   │
                                                    tour π  +  log p_θ(π|s)
                                                                   │
                          REINFORCE (§4):  ∇L = E[ (L(π) − b(s)) · ∇ log p_θ(π|s) ]
```

- **Encoder** (§3.1): 입력 feature 를 `d_h=128` 로 projection 후, **파라미터를 공유하지 않는
  N=3개 attention layer** 를 통과시킵니다. 각 layer = multi-head attention(M=8, `d_k=d_v=16`)
  + node-wise feed-forward(hidden 512, ReLU), 두 sublayer 모두 **skip-connection + batch
  normalization**(LayerNorm 아님 — §3.1 에서 BN 이 더 잘 동작). graph embedding 은 최종 node
  embedding 의 평균입니다. positional encoding 은 없습니다(순서 불변).
- **Decoder** (§3.2): 매 스텝 **context node** 를 만들고 → M-head *glimpse* 로 그래프를 한 번
  훑은 뒤 → single-head *pointer* 로 각 노드의 logit 을 계산합니다. logit 은 `C·tanh(·)`(C=10)
  로 클리핑한 **다음에** 방문/금지 노드를 `−∞` 로 마스킹합니다(클립과 마스크 순서가 중요 — §3.2).
  `greedy`(argmax) 또는 `sampling`(정책에서 추출)으로 노드를 고릅니다.
- **Training** (§4, Algorithm 1): REINFORCE 의 분산을 **greedy rollout baseline** `b(s)` 로
  줄입니다 — baseline 은 "지금까지 가장 좋은 모델"의 greedy 비용이며, 매 에폭 paired t-test(α=5%)
  로 현재 정책이 유의하게 더 나을 때만 교체합니다. 첫 에폭은 exponential baseline(β=0.8)으로 warmup.
  advantage `L(π) − b(s)` 는 상수 가중치이므로 detach 합니다.

## ✨ TSP 와 CVRP — 무엇이 같고 무엇이 다른가

인코더·glimpse·pointer·REINFORCE·baseline 은 **완전히 동일**합니다. 문제별로 바뀌는 부분만
`src/problems/<problem>/` 에 격리했습니다.

| 측면 | TSP | CVRP |
|------|-----|------|
| 노드 | n개 도시 | depot 1 + 고객 n |
| 입력 임베딩 | 좌표 `Linear(2, d_h)` | depot `Linear(2, d_h)` + 고객 `Linear(3, d_h)`(좌표+demand) |
| 디코더 context | `[graph, last, first]` (3·d_h) | `[graph, last, 잔여용량]` (2·d_h+1) |
| 마스크 | 방문한 노드 | 방문 고객 ∪ (demand > 잔여용량); depot 은 연속 방문 금지 |
| rollout 길이 | 정확히 n 스텝(해밀턴 경로) | **가변**(depot 재방문) — 모든 고객 방문 시 종료 |
| 비용 `L(π)` | 닫힌 tour 길이 | depot→첫노드 + 내부 + 마지막→depot |
| t=1 시작 | 학습된 placeholder `v^f, v^l` | depot 에서 시작(placeholder 불필요) |

## 🗂️ 구조와 확장성

```
src/
├── nets/                 # 문제 무관 코어 (TSP·CVRP 공유)
│   ├── config.py         # ModelConfig (d_h, N, M, d_ff, tanh_clip ...)
│   ├── layers.py         # MultiHeadAttention, AttentionLayer
│   ├── encoder.py        # Encoder (embedder 주입 가능)
│   └── decoder.py        # AttentionDecoder — State 가 구동하는 glimpse + pointer 루프
├── problems/
│   ├── state.py          # DecodeState 인터페이스
│   ├── tsp/              # state · decoder · data · cost
│   └── cvrp/            # embedder · state · decoder · data · cost
├── model.py              # AttentionModel 조립 + 문제 레지스트리(_PROBLEMS)
├── baseline.py · loss.py # greedy rollout baseline + t-test / REINFORCE loss
├── train.py · evaluate.py · report.py
└── configs/  notebooks/walkthrough.ipynb  REPRODUCTION_NOTES.md
```

새 라우팅 문제는 **디코더를 건드리지 않고** `DecodeState` 4개 메서드만 구현하면 추가됩니다.
디코더는 매 스텝 이 인터페이스에만 의존합니다:

```python
class DecodeState:
    def get_context(self): ...   # (batch, context_dim) — 현재 컨텍스트
    def get_mask(self):    ...   # (batch, n) bool — 선택 금지 노드
    def update(self, sel): ...   # 선택된 노드로 상태 전이
    def all_done(self):    ...   # 배치 전체 완료 여부 → rollout 종료
```

`all_done()` 으로 rollout 길이를 문제가 결정하므로, CVRP 의 **가변 길이**(depot 재방문)도
디코더 코드 수정 없이 처리됩니다.

## 📊 결과 (TSP-20 / CVRP-20)

축소 학습 버짓(20 epoch × 500 step, **논문 풀버짓의 약 1/25**)으로 단일 RTX 3060 Ti에서 수 시간
학습했습니다. 아키텍처·하이퍼파라미터는 논문 그대로이며 버짓만 줄였습니다. held-out 1,000개 고정
인스턴스 기준입니다.

| 문제 | 디코딩 | 길이 | optimality gap | 논문(풀버짓) |
|------|--------|------|----------------|--------------|
| **TSP-20** | greedy | 3.8853 | +1.45% | ≈3.85 |
| **TSP-20** | sampling (1280) | **3.8475** | **+0.46%** | ≈3.84 |
| **CVRP-20** | greedy | 6.6129 | +7.70% | ≈6.40 |
| **CVRP-20** | sampling (1280) | **6.2745** | **+2.19%** | ≈6.25 |

> gap 기준: TSP-20 optimal ≈3.83(Concorde), CVRP-20 ≈6.14(LKH3).

**해석.** 학습 곡선은 매끄럽게 수렴했습니다(TSP held-out greedy 10.4→3.89, CVRP 12.8→6.61).
버짓을 1/25 로 줄였음에도 **sampling 디코딩은 논문 풀버짓 수치에 근접**(TSP +0.46%, CVRP +2.19%)
하여, 모델·학습 루프·baseline·마스킹 구현이 정확함을 확인했습니다. 남은 greedy gap 은 학습량을
풀버짓으로 늘리면 좁혀집니다.

### CVRP — 고전 휴리스틱 baseline 과의 비교

학습 모델이 실제로 의미 있는지 보려면 **학습 없이 푸는 고전 휴리스틱**과 같은 조건에서 대조해야
합니다. Nearest Neighbor 와 Clarke-Wright Savings 를 `src/problems/cvrp/heuristics.py` 에
구현하고, **동일한 held-out 1,000 인스턴스·동일한 비용 함수**로 평가했습니다.

| 방법 | 종류 | CVRP-20 길이 | gap (vs LKH 6.14) |
|------|------|------|-------------------|
| Nearest Neighbor | 구성 휴리스틱 | 8.0318 | +30.81% |
| Attention Model — greedy | 학습 | 6.6129 | +7.70% |
| Clarke-Wright Savings | 구성 휴리스틱 | 6.3727 | +3.79% |
| **Attention Model — sampling (1280)** | 학습 | **6.2732** | **+2.17%** |
| (참고) LKH3 | 정확해에 가까움 | 6.1400 | — |

**해석.** 단순한 Nearest Neighbor 는 크게 뒤처지고(+30.8%), 잘 설계된 고전 휴리스틱
Clarke-Wright 는 **AM greedy 를 앞섭니다**. 하지만 AM 을 **sampling(1280)** 으로 디코딩하면
Clarke-Wright 까지 넘어서 LKH 에 가장 근접합니다. 즉 *"학습 모델 > 모든 고전 휴리스틱"* 이
무조건 성립하는 것이 아니라 **디코딩 전략에 달려 있다**는 점을 정직하게 보여줍니다.

```bash
python -m src.compare_baselines --config configs/cvrp_reduced.yaml --checkpoint outputs_cvrp/best.pt
```

## 🚀 빠른 시작

```bash
pip install -r requirements.txt

# 1) 학습 — held-out greedy 곡선 출력 + best.pt 저장
python -m src.train --config configs/reduced.yaml          # TSP-20
python -m src.train --config configs/cvrp_reduced.yaml     # CVRP-20

# 2) 평가표 — greedy + sampling(1280) + optimality gap
python -m src.report --config configs/reduced.yaml      --checkpoint outputs/best.pt
python -m src.report --config configs/cvrp_reduced.yaml --checkpoint outputs_cvrp/best.pt

# 3) 논문 풀버짓(100 epoch × 2500 step) 재현 — GPU 권장
python -m src.train --config configs/base.yaml
```

빠른 sanity check 는 [`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb) 에서 논문↔코드를
셀 단위로 따라가며 실행할 수 있습니다. 모든 하이퍼파라미터는 [`configs/`](configs) 에 있고 각 값에
논문 인용을 달았습니다.

## 🔬 재현 노트

논문이 명시한 값과 명시하지 않은 값(그리고 우리의 선택·근거)을 표로 정리했습니다 —
[`REPRODUCTION_NOTES.md`](REPRODUCTION_NOTES.md). 예: `d_k = d_h/M`(App. A 의 "M·d_h" 는 오식),
QKV bias 미사용, gradient clipping `max_norm=1.0`(공식 코드) 등.

## 📖 인용

```bibtex
@inproceedings{kool2019attention,
  title     = {Attention, Learn to Solve Routing Problems!},
  author    = {Kool, Wouter and van Hoof, Herke and Welling, Max},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2019},
  url       = {https://arxiv.org/abs/1803.08475}
}
```
