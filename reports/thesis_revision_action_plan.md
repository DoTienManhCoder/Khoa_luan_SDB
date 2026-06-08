# Ke Hoach Sua Khoa Luan Sau Khi Review PDF

Ngay lap: 2026-06-03

## 1. Muc Tieu Sua

Nhan xet tong quan danh gia khoa luan o muc kha tot den tot, nhung can siết lai ba diem truoc khi nop:

- Tinh tai lap cua thuc nghiem.
- Cach trinh bay threshold/validation/test de tranh hieu nham tuning tren test.
- Bang ablation tach rieng thanh phan.

Ngoai ra can sua loi hinh thuc nho o Hinh 4.3 va giam cac cum tu qua manh nhu "state-of-the-art" neu khong co benchmark day du voi moi baseline gan day.

## 2. Viec Nen Sua Truoc Khi Nop

### 2.1 Bo Sung Bang Moi Truong Chay

Them bang vao Chương 4, phan thiet lap thuc nghiem.

Noi dung de xuat:

Gia tri thuc te tren may local (cap nhat 2026-06-05; neu run tren Kaggle thi ghi them dong tuong ung):

| Hang muc | Gia tri |
|---|---|
| He dieu hanh | Windows 10 Pro 19045 (local) va Kaggle Linux tuy theo run |
| Python | 3.12.7 (Anaconda) |
| PyTorch | 2.5.1 (build cu121) |
| CUDA | 12.1 (neu dung GPU build) |
| GPU | Kaggle P100/T4 cho train; local CPU cho calibration/eval tu cache |
| Thu vien chinh | numpy 2.2.6, scipy 1.17.1, einops 0.8.2, opencv-python 4.12.0.88, pillow 10.4.0, matplotlib 3.10.8 |
| Seed | 42 |
| Checkpoint goc | `ckpt_0_200_0.pth` |
| Checkpoint Phase2 | `ckpt_phase2_shot_clipshots_best.pth` hoac checkpoint run tuong ung |
| Output chinh | `artifacts/experiments/ablation_full/`, `reports/postprocess_calibration_*.{json,md}` |

Cac version tren da duoc ghim trong `requirements.txt`.

Doan van de chen:

```text
Để bảo đảm khả năng tái lập, toàn bộ thí nghiệm được chạy với seed cố định 42. Các kết quả ablation sử dụng cùng split train/validation/test, cùng checkpoint AutoShot gốc và cùng metadata. Những artifact nặng như checkpoint, cached logits và video dataset không được đưa vào Git, mà được quản lý riêng theo manifest artifact.
```

### 2.2 Lam Ro Validation/Test Va Threshold

Them vao Chương 4, truoc bang ket qua chinh.

Doan van de chen:

```text
Ngưỡng quyết định không được chọn trực tiếp từ test khi báo cáo cấu hình deploy. Với các thí nghiệm Phase2 và post-process, threshold deploy được xác định từ validation và sau đó khóa lại khi đánh giá trên test. Riêng baseline AutoShot gốc (`A0_autoshot_original`) được báo cáo như một mốc tham chiếu lịch sử bằng best threshold sweep theo từng dataset; kết quả này không được dùng làm control trực tiếp cho ablation Phase2. Control chính của ablation là `A1_phase2_bce_onehot`.
```

Can tach bang:

- Bang "AutoShot goc voi best threshold" cho `A0`.
- Bang "Ablation tren control A1" cho cac run A1/A2/A3/P1/P2/B1/B4/B5.

### 2.3 Them Bang Ablation Tach Rieng

Dung ket qua da cap nhat tai:

```text
artifacts/experiments/ablation_full/ablation_results.csv
artifacts/experiments/ablation_full/ablation_results.json
```

Bang chinh de chen:

| Run | SHOT F1 | ClipShots F1 | BBC F1 | Ket luan |
|---|---:|---:|---:|---|
| `A0_autoshot_original` | 0.8405 | 0.7649 | 0.9554 | Baseline goc, best threshold |
| `A1_phase2_bce_onehot` | 0.8378 | 0.6983 | 0.9570 | Control Phase2 |
| `A2_focal_only` | 0.8378 | 0.6967 | 0.9567 | Focal rieng le khong cai thien |
| `A3_manyhot_only` | 0.8375 | 0.7005 | 0.9563 | Many-hot tang ClipShots nhe |
| `P1_gaussian_only` | 0.8432 | 0.7519 | 0.9436 | Tang ClipShots nhung fail BBC |
| `P2_temperature_only` | 0.8378 | 0.6983 | 0.9570 | Khong khac A1 o deploy |
| `B1_focal_manyhot` | 0.8384 | 0.7006 | 0.9559 | Cai thien nho |
| `B4_temperature_gaussian` | 0.8540 | 0.7441 | 0.9570 | Candidate deploy tot nhat |
| `B5_full_candidate_no_ema` | 0.8542 | 0.7409 | 0.9551 | Full khong EMA, thua B4 tren ClipShots/BBC |

Doan ket luan de chen:

```text
Kết quả ablation cho thấy các thành phần train-time như Focal Loss và many-hot target chỉ tạo cải thiện nhỏ hoặc không ổn định. Thành phần có tác động lớn nhất là hậu xử lý Gaussian smoothing. Tuy nhiên, Gaussian riêng lẻ làm giảm mạnh BBC, vì vậy cấu hình được chọn là `B4_temperature_gaussian`, tức kết hợp temperature scaling và Gaussian smoothing trên logits của control A1. Cấu hình này tăng SHOT và ClipShots rõ rệt, đồng thời giữ BBC gần như không đổi.
```

### 2.4 Sua Hinh 4.3

Loi duoc ghi nhan: duong noi doc di qua nhan "AutoShot".

Huong sua:

- Neu Hinh 4.3 la TikZ: dich node `AutoShot` sang trai/phai hoac dung `bend left`/`bend right` cho edge.
- Neu Hinh 4.3 la SVG/PNG: chinh file goc, tang khoang cach giua node va duong noi, sau do export lai.
- Kiem tra lai trang chua Hinh 4.3 bang render PDF.

Checklist sau khi sua:

- Nhan "AutoShot" khong bi duong noi de len.
- Caption van nam trong trang.
- Khong tao overfull hbox moi.

### 2.5 Giam Cach Dien Dat Qua Manh

Tim va sua cac cum:

- "state-of-the-art" -> "một baseline mạnh", "phương pháp nền tảng", "kết quả cạnh tranh".
- "mô hình mới" -> "phiên bản cải tiến tầng phân loại và hậu xử lý".
- "đạt kết quả vượt trội" -> "cải thiện trên SHOT và giữ ổn định trên BBC".

Doan dien dat de xuat:

```text
AutoShotV2 không thay đổi backbone AutoShot gốc, mà tập trung cải tiến tầng phân loại và hậu xử lý để phù hợp hơn với bối cảnh video ngắn. Do đó, đóng góp của khóa luận mang tính thực dụng: tăng khả năng hiệu chỉnh quyết định boundary và cải thiện trade-off precision/recall, thay vì đề xuất một kiến trúc hoàn toàn mới.
```

### 2.6 Lam Ro Muc Do Tai Lap Cua Hai Bang So

Khi trinh bay ket qua o Chuong 4, can noi ro hai bang so co hai muc do tai lap khac nhau, de tranh
hieu nham rang moi con so deu chay lai duoc tu dau.

Doan van de chen (ngay sau bang ket qua deploy, hoac trong phan thao luan ve tinh tai lap):

```text
Các kết quả trong khóa luận được trình bày ở hai mức độ tái lập. Bảng kết quả triển khai (deploy) trên SHOT, ClipShots và BBC được lưu dưới dạng file kết quả JSON đã chốt; do logits của lần chạy deploy không được đưa vào kho mã nguồn (theo quy ước chỉ quản lý source và report), bảng này được tái lập ở mức đọc lại kết quả đã lưu, hoặc chạy lại suy luận từ checkpoint deploy trên video (cần GPU). Ngược lại, bảng baseline AutoShot gốc (A0) và các cấu hình ablation/calibration (A1, B4, B5) được tái lập đầy đủ từ logits đã lưu kèm: lệnh kiểm tra tái lập tính lại F1 trực tiếp từ logits và khớp với số báo cáo trong sai số 2×10⁻³. Như vậy, các con số dùng để rút ra kết luận chính của khóa luận — vai trò của hậu xử lý và khoảng cách hiệu năng trên ClipShots — đều thuộc nhóm tái lập được từ logits, bảo đảm tính kiểm chứng của các nhận định.
```

Diem can luu y: phan biet nay bao ve tinh trung thuc, vi cac ket luan quan trong deu dua tren nhom so
tai lap-tu-logits (A0/B4/B5), khong phai nhom chi co JSON (bang deploy). `README.md` (muc "Muc do tai lap")
va `docs/ARTIFACTS_MANIFEST.md` da ghi ro phan biet nay; chi can nhac lai mot cau trong luan van.

## 3. Noi Dung Nen Dua Vao Ket Luan

Doan ket luan de xuat:

```text
Các thí nghiệm cho thấy AutoShotV2 cải thiện đáng kể trên SHOT khi sử dụng hậu xử lý temperature scaling kết hợp Gaussian smoothing. Trên ClipShots, cải thiện so với control Phase2 là rõ rệt, nhưng baseline AutoShot gốc với best threshold vẫn là một mốc mạnh. Điều này cho thấy domain generalization sang ClipShots chưa được giải quyết hoàn toàn bằng fine-tuning hiện tại. Trong phạm vi khóa luận, cấu hình hậu xử lý `B4_temperature_gaussian` là lựa chọn cân bằng nhất vì cải thiện SHOT và ClipShots trong khi vẫn giữ BBC trong guardrail.
```

### 3.1 Bang Chung Calibration Cross-Validation (Bo Sung)

De cung co nhan dinh ve ClipShots mot cach dinh luong va chong "tune tren test", da bo sung thi nghiem
calibration hau xu ly theo tung dataset bang 5-fold cross-validation (script `autoshotv2.postprocess_calibration`,
ket qua `reports/postprocess_calibration_summary.md`). Knob (temperature, sigma, threshold) duoc chon tren
fold calib va do tren fold giu lai.

| Model | SHOT | ClipShots | BBC |
|---|---:|---:|---:|
| A0 baseline (AutoShot goc) | 0.8443 | 0.7881 | 0.9538 |
| A1 Phase2 control | 0.8540 | 0.7575 | 0.9631 |
| B5 Phase2 full | 0.8539 | 0.7555 | 0.9584 |

Doan ket luan bo sung de xuat:

```text
Khi hiệu chỉnh hậu xử lý theo từng dataset bằng cross-validation (không tuning trên test), Phase2 vượt baseline AutoShot gốc trên SHOT (0.854 so với 0.844) và BBC (0.963 so với 0.954), nhưng vẫn thua trên ClipShots (0.757 so với 0.788). Đáng chú ý, calibration giúp chính baseline gốc tăng rõ trên ClipShots (0.765 lên 0.788), cho thấy phần lớn cải thiện trên miền này đến từ hậu xử lý chứ không từ fine-tuning. Vì calibration không bù được khoảng cách ClipShots, kết luận domain generalization sang ClipShots chưa được giải quyết bằng pipeline hiện tại là có cơ sở định lượng. Khoảng cách giữa cấu hình tune-trên-test và cross-validation rất nhỏ, cho thấy quá trình hiệu chỉnh ổn định.
```

## 4. Thu Tu Uu Tien

1. Chen bang moi truong chay va seed.
2. Chen giai thich validation/test/threshold.
3. Chen cau lam ro muc do tai lap cua hai bang so (muc 2.6).
4. Chen bang ablation tach rieng thanh phan.
5. Sua Hinh 4.3.
6. Giam cac cum "state-of-the-art" va dien dat dong gop khiem ton hon.
7. Build lai PDF va render lai cac trang ket qua, dac biet trang Hinh 4.3 va bang ablation.

## 5. Ghi Chu Ve Build

Theo review hien tai, PDF khong con loi citation/reference. Cac canh bao con lai gom:

- Overfull hbox nhe.
- Underfull hbox nhe.
- Font substitution.

Day la canh bao hinh thuc, khong phai loi build nghiem trong. Nen uu tien sua noi dung va Hinh 4.3 truoc, sau do moi toi uu cac canh bao layout nho.

