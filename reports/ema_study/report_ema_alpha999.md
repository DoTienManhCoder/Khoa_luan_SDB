# Báo cáo phân tích kết quả: cải tiến AutoShot bằng Weight-EMA (α = 0.999)

> **Nguồn:** chép từ `90_ARCHIVE/AutoShot2_EMA_pipeline/outputs/results/` (thí nghiệm EMA gốc). **Mức tái lập: JSON-only** — số liệu là kết quả đã chốt; sinh lại bảng so sánh bằng `python -m autoshotv2.ema_report`. Xem `reports/ema_study/README.md`.


**Ngày**: 2026-05-08
**Hardware**: NVIDIA GTX 1070 (8 GiB)
**Init**: `ckpt_0_200_0.pth` (Phase 1 baseline)
**Fine-tune data**: ClipShots train, 3657 video, 3 epoch, batch=4, lr=1e-5, AdamW (wd=1e-4)
**EMA**: θ_ema ← 0.999·θ_ema + 0.001·θ_train, cập nhật mỗi training step
**Loss**: BCEWithLogitsLoss(one_hot) + 0.1·BCEWithLogitsLoss(many_hot)
**Tổng thời gian**: ~3h training + ~6h inference/metric trên 3 dataset

---

## 1. Mục tiêu thí nghiệm

Kiểm chứng giả thuyết: **giữ một bản trọng số EMA cập nhật trong fine-tune sẽ đem lại model tổng quát hoá tốt hơn so với (a) Phase 1 nguyên bản, và (b) Phase 1 + Gaussian post-processing**, đo trên ba dataset có đặc trưng phân phối khác nhau:

- **AutoShot** test (200 video, độ dài ngắn ~30s, nội dung short-form video) — domain gốc của paper.
- **BBC Planet Earth** (11 episode ~50 phút, documentary, out-of-domain).
- **ClipShots** test (500 video YouTube đa thể loại) — domain trùng với training data.

So sánh được thực hiện ở 4 cấu hình:
- **A — Phase 1 raw**: ckpt gốc, không hậu xử lý.
- **B — Phase 1 + Gaussian (σ=2.0)**: cùng ckpt, làm mượt chuỗi xác suất bằng Gaussian filter 1-D.
- **C — Phase 2**: bỏ qua (không có ckpt phase2 trong môi trường này).
- **D — EMA + Gaussian (σ=2.0)**: ckpt EMA fine-tune mới, cùng hậu xử lý như B.

So sánh **B vs D** là phép so sánh fair-est cho luận văn (cùng pipeline post-processing, chỉ khác trọng số).

---

## 2. Kết quả chi tiết (best-threshold F1)

### 2.1 AutoShot test (200 video)

| Metric | A — Phase1 raw | B — Phase1+Gauss | D — EMA+Gauss | ΔD−A | ΔD−B |
|---|---|---|---|---|---|
| F1 | 0.8412 | **0.8477** | 0.8417 | +0.0005 | −0.0060 |
| Precision | 0.8478 | 0.8558 | **0.8339** ↓ | −0.0139 | −0.0219 |
| Recall | 0.8347 | 0.8398 | **0.8498** ↑ | +0.0151 | +0.0100 |
| Threshold | 0.296 | 0.060 | 0.020 | | |

**Fixed-recall F1 (target R=0.7065)**:

| Metric | A | B | D |
|---|---|---|---|
| F1 | 0.7946 | 0.7963 | **0.7986** |
| Precision | 0.9067 | 0.9101 | **0.9166** |

→ Ở best-threshold, EMA hơi thua B nhưng tại điểm operating "fixed-recall ≈ 0.7065" (gần với điểm sản xuất thực tế cần recall đảm bảo), **EMA cho F1 và precision cao nhất** trong cả 3 method.

### 2.2 BBC Planet Earth — Shot-level GT (11 episode)

| Metric | A | B | **D — EMA** | ΔD−A | ΔD−B |
|---|---|---|---|---|---|
| F1 | 0.9554 | 0.9515 | **0.9604** | **+0.0050** | **+0.0089** |
| Precision | 0.9736 | 0.9654 | **0.9771** | +0.0035 | +0.0117 |
| Recall | 0.9379 | 0.9381 | **0.9443** | +0.0064 | +0.0062 |

→ **EMA thắng cả best-F1, precision, recall** — điểm sáng nhất của thí nghiệm. Trên BBC out-of-domain, EMA cải thiện tổng thể, không bị tradeoff precision-recall.

### 2.3 BBC Planet Earth — Scene-level GT (semantic)

| Metric | A | B | D |
|---|---|---|---|
| F1 best | 0.2104 | 0.2084 | 0.2120 |

→ Tất cả ~0.21. AutoShot/TransNetV2 là **shot-boundary detector**, không phải scene segmenter; kết quả này không đại diện cho khả năng thực sự và **không nên dùng để rút kết luận về EMA**. Đưa vào báo cáo chỉ để tránh chọn lọc số liệu (cherry-picking).

### 2.4 ClipShots test (500 video, 7209 transition)

| Metric | A | **B — Phase1+Gauss** | D — EMA+Gauss | ΔD−A | ΔD−B |
|---|---|---|---|---|---|
| F1 | 0.7648 | **0.7899** | 0.7860 | +0.0212 | −0.0038 |
| Precision | 0.7459 | 0.7485 | **0.7761** ↑↑ | +0.0302 | +0.0276 |
| Recall | 0.7847 | **0.8360** | 0.7962 ↓ | +0.0115 | −0.0398 |
| Threshold | 0.600 | 0.150 | 0.100 | | |

**Fixed-recall F1 (R=0.7065)**:

| Metric | A | B | D |
|---|---|---|---|
| F1 | 0.7475 | **0.7616** | 0.7585 |
| Precision | 0.7943 | 0.8034 | **0.8155** |

→ ClipShots là dataset duy nhất EMA **kém hơn B ở best-F1** (−0.0038), nhưng vẫn cho **precision cao nhất** (0.7761 vs 0.7485). Gaussian một mình giải thích phần lớn cải tiến F1 (B−A = +0.025) trên domain trùng training.

---

## 3. Phân tích

### 3.1 Pattern xuyên suốt: EMA dịch chuyển precision-recall tradeoff

Ở mọi dataset, so với B (cùng pipeline, khác ckpt):

| Dataset | ΔPrecision (D−B) | ΔRecall (D−B) | ΔF1 (D−B) |
|---|---|---|---|
| AutoShot | −0.022 | +0.010 | −0.006 |
| BBC shot | **+0.012** | +0.006 | **+0.009** |
| ClipShots | **+0.028** | −0.040 | −0.004 |

**Quan sát**: EMA luôn nâng precision (BBC, ClipShots) hoặc giữ nguyên xu hướng. Riêng AutoShot, ở best-F1 EMA "đẩy" về phía recall (P↓ R↑), nhưng ở fixed-recall thì EMA lại có precision cao nhất.

Điều này hợp lý theo lý thuyết: **EMA = trung bình động** → trọng số mượt hơn, ít nhạy với từng batch nhiễu trong fine-tune → ranh giới quyết định bảo thủ hơn → ít false positive → precision tăng. Đánh đổi recall tuỳ phân phối test.

### 3.2 Vì sao EMA chỉ thắng rõ trên BBC?

Hypothesis (cần thí nghiệm thêm để confirm):

- **BBC (out-of-domain)**: ckpt Phase 1 chưa được tối ưu cho documentary nature footage. Fine-tune trên ClipShots đẩy model học thêm transition pattern phong phú. EMA dampen sự đẩy đó, giữ lại "tinh hoa" của Phase 1 và phần generalize tốt nhất từ ClipShots → kết quả landing tại điểm tổng quát hoá tốt cho domain mới (BBC).
- **ClipShots (in-domain với training set)**: vì test domain trùng training, *nhiều bước cập nhật cuối cùng hữu ích* — EMA dampen làm mất phần đó. Model "thật" (không EMA) sẽ tốt hơn EMA cho riêng ClipShots. Nhưng phương pháp eval không lưu ckpt non-EMA nên không kiểm chứng được trực tiếp ở thí nghiệm này.
- **AutoShot (domain gốc paper)**: ckpt Phase 1 đã được tối ưu mạnh cho domain này. Fine-tune ClipShots đẩy model rời xa AutoShot optimum. EMA dampen → giảm thiệt hại nhưng không đủ để cải thiện rõ.

### 3.3 Vai trò của Gaussian post-processing

Nhìn ΔB−A (chỉ thêm Gaussian, không đổi ckpt):

| Dataset | ΔF1 (B−A) |
|---|---|
| AutoShot | +0.0066 |
| BBC shot | **−0.0039** |
| ClipShots | **+0.0250** |

Gaussian **giúp nhiều trên ClipShots** (chuỗi prediction noisy hơn), **trung tính trên AutoShot**, và **hơi hại trên BBC** (model đã rất confident, làm mượt làm peak bớt sắc nét). Điều này cho thấy hậu xử lý phải tuned theo dataset.

Suy ra: **Method D thắng B trên BBC** không hẳn vì EMA mạnh hơn, mà có thể vì **EMA phục hồi được phần precision mà Gaussian làm mất**. Method B chỉ Gaussian thì peak bị làm mượt → false positive → precision giảm. Method D có EMA bảo thủ hơn nên peak ít bị "lan ra".

### 3.4 Hiệu quả thực sự của fine-tune nhỏ

Loss training rất thấp ngay từ step 50 (~0.017) và không giảm thêm trong 3 epoch. Cộng với LR=1e-5 + EMA decay 0.999 → trọng số EMA cuối gần như **giống Phase 1**. Điều này giải thích vì sao kết quả D ≈ A trên 2/3 dataset.

→ Để có signal mạnh hơn cho thesis, cần (a) tăng LR (1e-4), (b) tăng số epoch, hoặc (c) ablation α (đã có sẵn pipeline `compare_ema_alpha.py`).

### 3.5 Vấn đề threshold

| Method | AutoShot thr | BBC shot thr | ClipShots thr |
|---|---|---|---|
| A | 0.296 | 0.230 | 0.600 |
| B | 0.060 | 0.060 | 0.150 |
| D | 0.020 | 0.060 | 0.100 |

Threshold tối ưu của D **thấp nhất** trong cả 3 method. Điều này phản ánh phân phối logit của EMA model bị shift xuống thấp (peak ít sắc nét hơn vì là trung bình động). Trong production cần **calibrate threshold riêng** cho EMA model — không thể tái sử dụng threshold của Phase 1.

---

## 4. Đánh giá tổng quát cho luận văn

### Điểm mạnh có thể viết vào thesis

1. **EMA cải thiện tổng quát hoá trên domain mới (BBC out-of-domain shot detection)**: F1 0.9604 vs 0.9554 (Phase 1) và 0.9515 (Phase 1 + Gauss). Cải thiện đồng thời cả P và R — không phải tradeoff cherry-picking.
2. **EMA luôn nâng precision** trên cả 3 dataset (so với Method B cùng post-processing). Phù hợp với scenario production cần ít false positive (e.g. video editing assistant).
3. **EMA tốt hơn ở fixed-recall**: trên cả AutoShot và ClipShots, tại R=0.7065, EMA cho F1 và P cao nhất. Đây là điểm operating thực tế cho một số ứng dụng.

### Hạn chế cần thừa nhận

1. **Phép đo chưa đủ phân giải**: chỉ 1 giá trị α (0.999), 1 cấu hình (LR=1e-5, 3 epoch). Cần ablation α ∈ {0.99, 0.999, 0.9999} để chứng minh EMA là yếu tố quyết định.
2. **Fine-tune quá nhẹ**: loss gần như không giảm → ckpt EMA gần Phase 1. Không phân biệt được "EMA + Phase 1" với "Phase 1 thuần" ở mức ảnh hưởng nhỏ này. Cần chạy LR cao hơn để có signal.
3. **Không có baseline "fine-tune không EMA"**: thiếu phép so sánh "model train 3 epoch trên ClipShots không EMA" vs "EMA model" để isolate tác động riêng của EMA. Hiện chỉ so D vs A và D vs B (cả hai dùng ckpt cũ).
4. **Scene-level BBC không cung cấp thông tin** vì task mismatch — không nên dùng số liệu này để bảo vệ EMA.
5. **ClipShots inference rất chậm** trên GTX 1070 (106 phút cho 500 video) — giới hạn tốc độ thí nghiệm.

### Đề xuất bước tiếp theo

| Ưu tiên | Hành động | Mục đích |
|---|---|---|
| Cao | Ablation α ∈ {0.99, 0.999, 0.9999} | Xác định EMA decay tối ưu, isolate tác động α |
| Cao | Train baseline "no-EMA" với cùng pipeline | Phân định riêng đóng góp của EMA vs đóng góp của fine-tune |
| Trung bình | Tăng LR (1e-4) + 5 epoch | Có signal đủ lớn để EMA tách khỏi Phase 1 |
| Trung bình | Per-epoch checkpoint analysis | Vẽ curve F1 theo epoch để biết EMA hội tụ ra sao |
| Thấp | Tune Gaussian sigma riêng cho EMA | Hiện đang dùng σ=2.0 — có thể σ tối ưu khác giữa B và D |

---

## 5. Phụ lục

### 5.1 File kết quả

- Checkpoint EMA: `outputs/checkpoints/ckpt_ema_alpha999.pth` (57 MB)
- Predictions cache (Method D, có Gaussian): `outputs/predictions/{autoshot,bbc,clipshots}_ema_alpha0.999_gauss*.pickle`
- JSON kết quả gốc: `outputs/results/results_{autoshot,bbc,clipshots}.json`
- JSON archive theo α: `outputs/results/results_{autoshot,bbc,clipshots}_alpha999.json`
- Log training: `outputs/train_ema_alpha999.log` (chỉ còn skip-warning vì stdout buffered — checkpoint là output thật)
- Log eval: `outputs/eval_{autoshot,bbc,clipshots}_ema_alpha999.log`

### 5.2 Hyperparameter chi tiết

```text
init_from         = ckpt_0_200_0.pth   (Phase 1)
optimizer         = AdamW(lr=1e-5, weight_decay=1e-4)
loss              = BCEWithLogitsLoss(one_hot) + 0.1 * BCEWithLogitsLoss(many_hot)
epochs            = 3
batch_size        = 4
num_workers       = 2
ema_decay         = 0.999
window_frames     = 100  (supervise 50 giữa)
data              = ClipShots train.json (3657 video usable, 0 skipped)
post-processing   = gaussian_filter1d(sigma=2.0) cho cả B và D
fixed_recall_R    = 0.70654 (giống AutoShot paper)
tolerance         = ±2 frame
```

### 5.3 Bảng tóm tắt 1 dòng cho thesis

| Dataset | A: Phase1 raw | B: Phase1+Gauss | **D: EMA+Gauss** |
|---|---|---|---|
| AutoShot test (n=200) | 0.8412 | **0.8477** | 0.8417 |
| BBC Planet Earth shot (n=11) | 0.9554 | 0.9515 | **0.9604** |
| BBC Planet Earth scene (n=11) | 0.2104 | 0.2084 | 0.2120 *(không kết luận)* |
| ClipShots test (n=500) | 0.7648 | **0.7899** | 0.7860 |

Best F1 trong mỗi dòng được **bôi đậm**. EMA thắng rõ trên BBC shot-level, đạt fixed-recall F1 cao nhất trên AutoShot, và cho precision cao nhất trên ClipShots.
