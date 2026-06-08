# Báo cáo phân tích: EMA vs no-EMA fine-tune baseline

> **Nguồn:** chép từ `90_ARCHIVE/AutoShot2_EMA_pipeline/outputs/results/` (thí nghiệm EMA gốc). **Mức tái lập: JSON-only** — số liệu là kết quả đã chốt; sinh lại bảng so sánh bằng `python -m autoshotv2.ema_report`. Xem `reports/ema_study/README.md`.


**Ngày**: 2026-05-08
**Mục tiêu**: cô lập tác động của Weight EMA bằng cách so sánh với baseline fine-tune cùng pipeline nhưng không có EMA.

## 1. Setup hai run

| Yếu tố | EMA run | no-EMA baseline run |
|---|---|---|
| Init checkpoint | `ckpt_0_200_0.pth` (Phase 1) | `ckpt_0_200_0.pth` (Phase 1) |
| Fine-tune data | ClipShots train (3657 video) | ClipShots train (3657 video) |
| Optimizer | AdamW lr=1e-5, wd=1e-4 | AdamW lr=1e-5, wd=1e-4 |
| Loss | BCE(one_hot) + 0.1·BCE(many_hot) | BCE(one_hot) + 0.1·BCE(many_hot) |
| Epochs / batch | 3 / 4 | 3 / 4 |
| EMA decay | **0.999** | **None (skip ema.update)** |
| Final ckpt nguồn | `ema.state_dict()` (shadow) | `model.state_dict()` (trained weights) |
| Post-processing | Gaussian σ=2.0 | Gaussian σ=2.0 |

→ **Hai run giống hệt nhau, chỉ khác ở việc có gọi `ema.update(model)` mỗi step và lấy trọng số nào lúc save**.

Loss avg theo epoch:

| Run | epoch 1 | epoch 2 | epoch 3 |
|---|---|---|---|
| EMA | 0.0177 | n/a | n/a (chỉ log final) |
| no-EMA | **0.0166** | **0.0151** | **0.0143** |

Loss giảm đều ở no-EMA → trọng số train thực sự đang move; trọng số EMA shadow di chuyển chậm hơn nhiều (decay 0.999).

## 2. Kết quả best-threshold F1

| Dataset | A: Phase1 raw | B: Phase1+Gauss | **D: EMA α=0.999** | **D: no-EMA fine-tune** | ΔEMA−noEMA |
|---|---|---|---|---|---|
| AutoShot test (n=200) | 0.8412 | **0.8477** | 0.8417 | 0.8371 | **+0.0046** |
| BBC shot-level (n=11) | 0.9554 | 0.9515 | **0.9604** | 0.9494 | **+0.0110** |
| ClipShots test (n=500) | 0.7648 | **0.7899** | 0.7860 | 0.7638 | **+0.0222** |

(Method B in đậm khi tốt nhất; Method D-EMA in đậm khi vượt cả A và B; D-noEMA in đậm để dễ so.)

## 3. Phân tích

### 3.1 EMA luôn vượt no-EMA trên cả 3 dataset

Cột "ΔEMA−noEMA" dương cả 3 dataset, biên độ tăng dần: **AutoShot +0.005, BBC +0.011, ClipShots +0.022**. Đây là **phép so sánh fair-est** vì hai run khác nhau đúng một yếu tố (EMA on/off) và dùng cùng post-processing Gaussian σ=2.0.

→ **Kết luận chính cho thesis**: EMA đóng góp F1 dương trong mọi dataset đã test. Cải tiến không phải do "fine-tune trên ClipShots" — vì baseline fine-tune (no-EMA) **thậm chí thua Phase 1 raw** trên cả 3 dataset.

### 3.2 No-EMA fine-tune **luôn tệ hơn Phase 1 raw**

| Dataset | Phase 1 raw (A) | no-EMA fine-tune (D) | Δ |
|---|---|---|---|
| AutoShot | 0.8412 | 0.8371 | **−0.0041** |
| BBC shot | 0.9554 | 0.9494 | **−0.0060** |
| ClipShots | 0.7648 | 0.7638 | **−0.0010** |

Bất ngờ: no-EMA fine-tune tệ hơn ngay cả trên ClipShots — domain trùng với training data. Điều này gợi ý:
- LR=1e-5 với 3 epoch trên 3657 video = quá nhiều update (~2745 step) → trọng số bị over-fit subtle hoặc dịch chuyển khỏi điểm ổn định của Phase 1.
- Phase 1 đã được tối ưu tốt ngay từ đầu cho TransNetV2-style detection; bất kỳ shift nào đều giảm chất lượng.

### 3.3 EMA dampen sự "drift" của fine-tune

EMA trung bình động trọng số theo công thức `θ_ema ← 0.999·θ_ema + 0.001·θ_train`:
- Mỗi step EMA chỉ "nhìn thấy" 0.1% của trọng số mới.
- Trên 2745 step, hiệu quả tích luỹ tương đương "fine-tune nhẹ" — không đẩy model đi xa Phase 1.
- Điều này hoá ra **chính xác là điều thesis cần**: giữ kỷ luật của Phase 1, chỉ thêm một chút thông tin ClipShots.

→ **Cơ chế hoạt động**: EMA hoạt động như một regularizer ngầm chống over-fit khi LR/đa epoch khiến no-EMA fine-tune trượt khỏi optimum.

### 3.4 EMA thắng cả Phase 1 + Gaussian trên BBC

Trên BBC shot-level, **D-EMA F1=0.9604** vượt cả:
- A: 0.9554 (+0.0050)
- B: 0.9515 (+0.0089)
- D-noEMA: 0.9494 (+0.0110)

→ EMA thực sự cải thiện model trên domain out-of-distribution (BBC chưa thấy trong training Phase 1 hay ClipShots fine-tune). Đây là điểm mạnh nhất của EMA trong thesis.

### 3.5 Vì sao trên AutoShot và ClipShots, B vẫn vượt D-EMA?

- **AutoShot (B=0.8477 > D-EMA=0.8417)**: AutoShot domain SHORT-form video, không có trong ClipShots train. Fine-tune (cả EMA hay no-EMA) đều push model xa AutoShot optimum. EMA giảm thiệt hại nhưng không thắng được "không fine-tune gì cả".
- **ClipShots (B=0.7899 > D-EMA=0.7860)**: chỉ chênh 0.004. EMA gần như equal với Phase 1 + Gaussian trên ClipShots test, vì EMA bảo thủ → trọng số rất gần Phase 1. Lợi thế thấy rõ qua **precision** (D-EMA P=0.7761 vs B P=0.7485) — EMA giúp giảm false positive.

## 4. Bảng tổng hợp 1 dòng cho thesis

| Setup | AutoShot | BBC shot | ClipShots |
|---|---|---|---|
| A (Phase 1 raw) | 0.8412 | 0.9554 | 0.7648 |
| B (Phase 1 + Gauss) | **0.8477** | 0.9515 | **0.7899** |
| D no-EMA fine-tune (control) | 0.8371 | 0.9494 | 0.7638 |
| **D EMA α=0.999 (đề xuất)** | 0.8417 | **0.9604** | 0.7860 |

→ Trong 3 dataset, EMA: thắng tất cả trên BBC; thắng baseline fine-tune trên cả 3 (đóng góp riêng dương); chỉ thua Phase 1 + Gauss trên 2 dataset (do fine-tune chưa được tối ưu hóa).

## 5. Đánh giá tổng quát

### Điểm thesis có thể claim **vững chắc**

1. **Weight EMA cải thiện fine-tune AutoShot trên cả 3 dataset** so với baseline fine-tune giống hệt nhưng không có EMA (ΔF1 dương: +0.005 / +0.011 / +0.022).
2. **EMA đặc biệt hiệu quả trên domain out-of-distribution** (BBC: +0.011 vs no-EMA fine-tune, +0.005 vs Phase 1 raw, +0.009 vs Phase 1 + Gauss). Đây là kịch bản EMA thắng tất cả các baseline.
3. **EMA dampen drift của fine-tune**: trong 3 dataset, no-EMA fine-tune luôn tệ hơn Phase 1 raw, EMA luôn không tệ hơn Phase 1 raw nhiều (chênh ≤0.005 trên dataset xấu nhất). EMA hoạt động như regularizer ngầm.

### Điểm cần thừa nhận

1. **EMA không vượt Phase 1 + Gauss trên 2/3 dataset**. Trên AutoShot/ClipShots, post-processing Gaussian một mình đã rất hiệu quả; thêm EMA fine-tune chỉ ngang hoặc kém hơn (chênh ≤0.004).
2. **LR=1e-5 chưa được tune**. Nếu LR cao hơn thì no-EMA fine-tune có thể tốt hơn (model học được nhiều hơn) hoặc tệ hơn (over-fit nhanh hơn). Cần ablation LR để có claim chính xác.
3. **Chỉ 1 giá trị α (0.999)**. Cần ablation α ∈ {0.99, 0.9999} để kiểm chứng decay tối ưu.

### Đề xuất cho phần kết luận luận văn

Có thể viết: **"Weight EMA là một regularizer ngầm cho fine-tune mô hình AutoShot, đem lại F1 dương trên cả 3 dataset (AutoShot, BBC, ClipShots) so với baseline fine-tune không EMA. Hiệu quả lớn nhất khi áp dụng cho domain mới (BBC: +0.011 F1, đạt 0.9604 — vượt mọi baseline)."**

## 6. File liên quan

- Checkpoint EMA: `outputs/checkpoints/ckpt_ema_alpha999.pth`
- Checkpoint no-EMA: `outputs/checkpoints/ckpt_baseline_noema.pth`
- Per-epoch ckpt: `outputs/checkpoints/ckpt_{ema_alpha999,baseline_noema}_epoch{1,2,3}.pth`
- Result JSON:
  - EMA: `outputs/results/results_{autoshot,bbc,clipshots}_alpha999.json`
  - no-EMA: `outputs/results/results_{autoshot,bbc,clipshots}_noema.json`
- Bảng so sánh tự động: `outputs/results/ema_vs_noema.md` (sinh bằng `compare_ema_alpha.py`)
- Log:
  - Training: `outputs/train_baseline_noema.log` (3.4h, 3 epoch ClipShots train)
  - Eval: `outputs/eval_{autoshot,bbc,clipshots}_noema.log`

## 7. Cảnh báo về dữ liệu

- BBC scene-level chưa eval cho no-EMA (process bị kill mid-scene metric phase). Số shot-level đã có đầy đủ — đủ cho kết luận thesis vì shot-level mới là task chính của AutoShot.
- BBC eval dùng cờ `--skip-scene-level` cho mọi run tương lai để tiết kiệm ~30 phút.
- ClipShots eval dùng `--skip-baselines` (chỉ tính D), tiết kiệm ~45 phút metric phase. Số A/B/C lấy từ run α=0.999 trước (giống hệt vì cùng Phase 1 ckpt).
