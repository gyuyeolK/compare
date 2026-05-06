# Option C — Quick Command Card

## The exact command to run on your H100

```bash
cd /path/to/optim_compare/

CUDA_VISIBLE_DEVICES=0 python train_1p3b.py \
    --data fineweb \
    --d_model 2048 --n_layers 24 --n_heads 32 \
    --seq_len 1024 --micro_batch 8 --grad_accum 256 \
    --steps 3000 --log_every 100 \
    --lr 0.01 --adam_lr 3e-4 --weight_decay 0.01 \
    --warmup_frac 0.0 --cooldown_frac 0.1 \
    --rank_fraction 0.25 --qr_method qr --qr_warmup_steps 200 \
    --opts dion \
    --track_kappa \
    --kappa_method svd \
    --kappa_stride 100 \
    --kappa_dense_until 200 \
    --out_dir results/dion_1p3b_kappa \
    2>&1 | tee results/dion_1p3b_kappa.log
```

`tee` 로 stdout 까지 백업해두면 학습 중 상태 확인 + ETA 추적 편함.

## 무엇이 일어나는가

| | |
|-|-|
| Model | 1.4B params (d=2048, 24L, 32H) |
| Tokens/step | 2.1M (micro_batch 8 × seq 1024 × accum 256) |
| Total tokens | 6.3B (3000 × 2.1M) |
| Wall-clock | ~38 hours on a single H100 (~1.6 days) |
| Memory | ~13 GB optimizer + ~1.5 GB activations on H100 80GB |
| κ overhead | ~22 sec (negligible vs 38h baseline) |

## 진행 모니터링

학습 중 stdout 형식:
```
  step  1500/3000 | train 4.8123 | val 4.6982 | lr_mult 1.000 | 68400s | ETA 0.5d | kappa_obs=14400
```

- `kappa_obs` 가 step 200 까지 step 당 96 씩 증가 (96 matrices × dense window)
- step 200 이후 100 step 마다 96 씩 증가 (sparse stride)
- step 3000 시점 expected: 200×96 + 28×96 = 21,888 observations

## 끝나면 보내주실 파일들

```
results/dion_1p3b_kappa/
├── history.json          # training curves (loss, wall-time)
├── kappa_log.json        # full per-matrix κ trajectory (~2 MB)
├── kappa_summary.json    # aggregate statistics
└── kappa_trajectory.png  # auto-generated plot
```

이 4개 파일을 받으면 paper Section 6.5 에:

1. **162M / 1.3B 비교 테이블** (post-warmup max, p99, p95)
2. **2-panel kappa plot** (162M 옆에 1.3B 추가)
3. **Section 6.7 Limitations 항목** "Model scale below frontier" 해소 또는 nuance 추가
4. **Discussion Section 7.1** "What is and is not proven" 의 κ 항목 강화

## 잠재 이슈 대비 (probably won't happen but...)

1. **OOM**: `--micro_batch 4` 로 줄이고 `--grad_accum 512` 로 늘리면 메모리 절반. 또는 `--activation_checkpointing` 추가. 1.3B + H100 80GB 에서는 거의 발생 안 함.

2. **느린 학습**: `nvidia-smi` 로 GPU utilization 체크. `~85%+` 이어야 함. 낮으면 dataloader bottleneck (FineWeb streaming) 가능 — `--micro_batch 16` 으로 늘리고 `--grad_accum 128` 로 줄이면 throughput 개선.

3. **κ tracking이 NaN/Inf**: 162M 에서 첫 2-3 step 의 pre-warmup 에서 그런 일이 있었음 (svd 측정도 numerical limit 도달). Logger 가 자동으로 `null` 또는 `"inf"` 로 저장하고 학습은 영향 안 받음. summary 의 `n_total_observations` 가 expected 22K 보다 약간 적게 나올 수 있음.

4. **학습 중 loss spike**: lr=0.01 + qr_method=qr 는 안전하지만, 만약 spike 발생하면 paper 의 reproducibility note 에서 `qr_warmup_steps` 가 plain QR 만 쓰는 상황이라 발생하지 않을 것.

## 시작 전 마지막 체크

```bash
# 1. 코드 sync
ls optimizers.py model.py data.py train_compare.py train_1p3b.py
# 2. requirements
pip list | grep -E "torch|datasets|transformers|matplotlib"
# 3. quick smoke test (몇 초 안에 끝남)
python train_1p3b.py --data synthetic --steps 2 \
    --d_model 256 --n_layers 2 --n_heads 4 --vocab_size 256 \
    --seq_len 32 --micro_batch 2 --grad_accum 2 \
    --opts dion --track_kappa \
    --out_dir /tmp/check && echo "smoke test passed"
```

smoke test가 통과하면 위 main command 시작.

## 결과 도착 후 paper 통합 절차

결과 파일 4개를 다시 보내주시면 즉시:
1. Section 6.5 Table 6 update (162M / 1.3B 두 컬럼)
2. Figure 1 update or 추가 (162M 옆에 1.3B trajectory)
3. Section 6.7 Limitations 의 "Model scale below frontier" 항목 정확한 measured value로 교체
4. Abstract / Contributions list 업데이트 (paper 의 main empirical claim 강화)
5. NeurIPS 26-page PDF 재컴파일

총 paper update 시간 약 20-30분 예상.
