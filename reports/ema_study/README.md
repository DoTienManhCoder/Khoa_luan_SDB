# Nghiên cứu EMA (fine-tune toàn model trên ClipShots)

Thí nghiệm này (gốc ở `90_ARCHIVE/AutoShot2_EMA_pipeline`) cô lập tác động của **Weight EMA**
khi fine-tune **toàn bộ** model AutoShot trên ClipShots train, bằng cách so với một control
fine-tune **giống hệt nhưng tắt EMA**.

> **Khác chế độ train với Phase2.** Phase2 (`autoshotv2.train_phase2`) chỉ train *classification head*
> trên feature cache (backbone đóng băng), EMA head-only. Nghiên cứu này (`autoshotv2.train_ema`) fine-tune
> *toàn model* trên video ClipShots, EMA toàn model (`autoshotv2.ema.ModelEMA`). Hai bộ số **không so trực tiếp**
> với bảng deploy/ablation Phase2; baseline A/B ở đây gần như trùng A0 của repo
> (ClipShots 0.7648 vs 0.7649; BBC 0.9554 trùng) nên dùng chung mốc tham chiếu được.

**Mức tái lập: JSON-only.** Số liệu là kết quả đã chốt trong `results_*.json`. Sinh lại bảng dưới đây bằng:

```powershell
python -m autoshotv2.ema_report
```

Checkpoint EMA/noEMA, prediction pickle và log **không kèm bundle** (xem `docs/ARTIFACTS_MANIFEST.md` mục 6).
Phân tích đầy đủ: [`report_ema_vs_noema.md`](report_ema_vs_noema.md) và [`report_ema_alpha999.md`](report_ema_alpha999.md).

4 phương pháp: **A** = Phase 1 raw · **B** = Phase 1 + Gaussian (σ=2.0) · **D-EMA** = fine-tune + EMA α=0.999 ·
**D-noEMA** = fine-tune giống hệt, tắt EMA (control).

## Best-threshold F1 / Precision / Recall

| Dataset | Method | F1 | Precision | Recall | thr |
|---|---|---:|---:|---:|---:|
| **AutoShot test** (n=200) | A — Phase 1 raw | 0.8412 | 0.8478 | 0.8347 | 0.296 |
| | **B — Phase 1 + Gauss** | **0.8477** | 0.8558 | 0.8398 | 0.06 |
| | D-EMA (α=0.999) | 0.8417 | 0.8339 | 0.8498 | 0.02 |
| | D-noEMA (control) | 0.8371 | 0.8449 | 0.8295 | 0.02 |
| **BBC shot-level** (n=11) | A — Phase 1 raw | 0.9554 | 0.9736 | 0.9379 | 0.23 |
| | B — Phase 1 + Gauss | 0.9515 | 0.9654 | 0.9381 | 0.06 |
| | **D-EMA (α=0.999)** | **0.9604** | 0.9771 | 0.9443 | 0.06 |
| | D-noEMA (control) | 0.9494 | 0.9758 | 0.9244 | 0.06 |
| **ClipShots test** (n=500) | A — Phase 1 raw | 0.7648 | 0.7459 | 0.7847 | 0.60 |
| | **B — Phase 1 + Gauss** | **0.7899** | 0.7485 | 0.8360 | 0.15 |
| | D-EMA (α=0.999) | 0.7860 | 0.7761 | 0.7962 | 0.10 |
| | D-noEMA (control) | 0.7638 | 0.7301 | 0.8008 | 0.06 |

## Đóng góp riêng của EMA (D-EMA − D-noEMA)

| Dataset | D-EMA F1 | D-noEMA F1 | ΔF1 |
|---|---:|---:|---:|
| AutoShot | 0.8417 | 0.8371 | **+0.0046** |
| BBC shot | 0.9604 | 0.9494 | **+0.0110** |
| ClipShots | 0.7860 | 0.7638 | **+0.0222** |

## Kết luận & caveat (trung thực)

- **Claim vững:** EMA cho ΔF1 dương trên **cả 3 dataset** so với control no-EMA, và **vượt mọi baseline trên BBC**
  (out-of-distribution) — EMA hoạt động như regularizer ngầm chống drift của fine-tune.
- **Cần thừa nhận:** EMA **thua Phase 1 + Gauss trên AutoShot và ClipShots** (chênh ≤ 0.004). Tức hậu xử lý
  Gaussian một mình đã rất mạnh; thêm fine-tune + EMA chỉ ngang hoặc kém ở 2/3 dataset.
- **Hạn chế:** chỉ thử **một** giá trị α = 0.999; **LR = 1e-5 chưa tune**; no-EMA fine-tune còn **tệ hơn Phase 1 raw**
  trên cả 3 dataset (gợi ý fine-tune hiện tại chưa tối ưu).
- BBC **scene-level** chỉ ~0.21 F1 (task khác shot-level) và **no-EMA chưa eval scene-level** (process bị kill) —
  không dùng làm số chính.
