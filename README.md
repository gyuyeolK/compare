# Muon vs Dion vs Dion2 — 비교 실험 코드

세 옵티마이저를 동일한 GPT-style 모델에서 학습 곡선과 step time으로 비교합니다.

## 구성

```
optimizers.py    Muon, Dion, Dion2 단일 GPU 구현 + AdamW와의 파라미터 분리 헬퍼
model.py         RoPE + RMSNorm + squared-ReLU MLP 의 소형 GPT (no-bias)
data.py          FineWeb-Edu 스트리밍 로더 + 합성 데이터 fallback
train_compare.py 세 옵티마이저로 같은 모델을 학습하고 loss 기록
benchmark.py     Figure 1 스타일의 step time 벤치마크
plot.py          기록된 loss curve 시각화
```

## 빠른 시작 (네트워크 없는 환경, smoke test)

```bash
pip install torch matplotlib
python train_compare.py --data synthetic --steps 200 --d_model 256 --n_layers 4
python plot.py
```

합성 데이터에서는 absolute loss 값에 의미가 없습니다 — 코드와 옵티마이저가 살아있는지 확인용입니다.

## FineWeb-Edu 로 진짜 비교 (논문 설정에 가깝게)

```bash
pip install torch matplotlib datasets transformers

# 160M 모델, ~Chinchilla optimal에 훨씬 못 미치지만 비교는 가능
python train_compare.py \
    --data fineweb \
    --d_model 768 --n_layers 12 --n_heads 12 \
    --batch_size 8 --seq_len 1024 \
    --steps 3000 --log_every 100 \
    --lr 0.02 --adam_lr 3e-4 \
    --rank_fraction 0.25 --alpha 0.25 --selection l1
```

3개 옵티마이저가 차례대로 학습되고, `results/history.json` 에 기록되어 plot으로 비교 가능합니다.

## Step time 벤치마크 (Figure 1 재현)

```bash
python benchmark.py --sizes 2048 4096 8192 --rank_fractions 0.25 0.0625 --alphas 0.5 0.25
```

## 하이퍼파라미터 노트 (논문 기준)

- **Muon**: lr 0.02, momentum 0.95, Nesterov, NS 5 step
- **Dion**: lr 0.01, rank_fraction ∈ {1/2, 1/4, 1/16}, error-feedback β=0.05
- **Dion2**: lr 0.02, alpha ∈ {0.5, 0.25, 0.125}, selection ∈ {l1, random}, momentum_decay 0.95
- 매트릭스가 아닌 파라미터(임베딩, LM head, RMSNorm scale, bias)는 모두 AdamW로 학습합니다 (논문의 표준 관행).
- per-matrix 학습률 스케일 √(fan_out / fan_in) 적용 — Dion 논문 Table 2의 Spectral norm 행을 따릅니다.

## 알고 있어야 할 단순화 / 한계

1. **단일 GPU·언샤드만 구현**. Dion 논문의 1D/2D 샤딩 알고리즘(Algorithm 3, 4)이나 Dion의 compressed DP-sync, Lazy-Dion / CPU-Dion 변형은 포함되어 있지 않습니다 — 분산 환경에서의 진짜 wall-clock 우위는 여기서 측정되지 않습니다.
2. **Dion의 power iteration**은 Algorithm 2의 single-step 형태입니다 (이전 V로 warm start). RCQR이나 Cholesky-QR 같은 Appendix A의 가속 변형은 사용하지 않고, 그냥 `torch.linalg.qr` 만 씁니다.
3. **Newton-Schulz**는 bf16 quintic iteration (계수 3.4445, -4.7750, 2.0315) 으로 GPU에 한해 빠릅니다. CPU에서는 fp32로 동작합니다.
4. **하이퍼파라미터 스윕**은 포함하지 않았습니다 — 동일 lr 0.02 (matrix) / 3e-4 (Adam) 가 세 옵티마이저 모두에 합리적이라는 것이 두 논문에서 확인된 출발점입니다. 작은 모델에서는 더 튜닝하면 결과가 달라질 수 있습니다.
5. **합성 데이터**로 비교하면 옵티마이저 사이에 의미 있는 차이가 거의 보이지 않습니다 — 토큰 분포가 너무 단순해서 매트릭스 업데이트의 질이 거의 영향을 안 줍니다. 비교는 FineWeb 같은 실제 텍스트로 해야 합니다.
