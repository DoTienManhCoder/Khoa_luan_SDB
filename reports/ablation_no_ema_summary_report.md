# Bao Cao Tong Hop Ablation AutoShotV2 Khong EMA

Ngay tong hop: 2026-06-03

> **Luu y:** Cac so trong bao cao nay thuoc pipeline ablation CO KIEM SOAT (control A1 +
> post-process, metadata subset "local", temperature = 0.661785). Day KHONG phai cung pipeline voi
> bang "Ket Qua Da Chay" trong `README.md` (checkpoint deploy, train full Shot+ClipShots,
> temperature = 0.38780). Hai bo so dung training data / temperature khac nhau nen khong so truc tiep;
> SHOT gan nhu trung (0.8540 vs 0.8545) la dau hieu nhat quan.

## 1. Muc Tieu

Muc tieu cua thuc nghiem la danh gia rieng tung thanh phan cai tien cho AutoShotV2 ma khong su dung EMA trong matrix chinh. Cac thanh phan duoc kiem tra gom:

- Phase2 training toi gian voi BCE va one-hot target.
- Focal Loss.
- Many-hot auxiliary target.
- Gaussian smoothing.
- Temperature scaling.
- To hop Focal + many-hot.
- To hop Temperature + Gaussian.
- Full candidate khong EMA: Focal + many-hot + temperature + Gaussian.

Ba dataset duoc bao cao:

- `SHOT`: dataset chinh de kiem tra kha nang giu chat luong tren mien AutoShot.
- `ClipShots`: dataset uu tien cai thien.
- `BBC`: guardrail generalization, chi co 11 video nen khong dung lam dataset chon hyperparameter chinh.

## 2. Luu Y Ve Cach Doc Metric

`A0_autoshot_original` la baseline AutoShot goc va duoc bao cao bang best threshold sweep rieng cho tung dataset. Cac run ablation con lai bao cao theo cau hinh deploy cua runner, voi control chinh la `A1_phase2_bce_onehot`.

Vi vay khong nen so sanh A0 va A1 nhu hai run co cung protocol deploy. A0 dung de tra loi cau hoi: checkpoint AutoShot goc co the dat toi dau neu sweep threshold rieng theo dataset. A1 dung lam control de do tac dong cua tung thanh phan trong pipeline Phase2/post-process.

## 3. Cau Hinh Thuc Nghiem

Thu muc output chinh:

```text
artifacts/experiments/ablation_full/
```

Output da cap nhat:

```text
artifacts/experiments/ablation_full/ablation_results.csv
artifacts/experiments/ablation_full/ablation_results.json
artifacts/experiments/ablation_full/ablation_summary.md
artifacts/experiments/ablation_full/figures/component_delta_f1.png
artifacts/experiments/ablation_full/figures/dataset_tradeoff.png
artifacts/experiments/ablation_full/figures/precision_recall_delta.png
```

Matrix mac dinh da bo EMA:

```text
A0_autoshot_original
A1_phase2_bce_onehot
A2_focal_only
A3_manyhot_only
P1_gaussian_only
P2_temperature_only
B1_focal_manyhot
B4_temperature_gaussian
B5_full_candidate
```

Sau khi bo EMA, cac run EMA cu (`A4_ema_only`, `B2_focal_ema`, `B3_manyhot_ema`) khong nam trong matrix chinh va khong duoc dung de chon deploy.

> **Cross-ref:** EMA duoc nghien cuu rieng trong `reports/ema_study/` (che do train KHAC: fine-tune toan model
> tren ClipShots + EMA toan model, khong phai train head tren cache nhu o day). Ket qua o do cho thay EMA
> co dong gop F1 duong so voi control no-EMA tren ca 3 dataset, nhung van thua Phase 1 + Gaussian tren
> AutoShot/ClipShots. Hai noi bo tro nhau, khong mau thuan: matrix Phase2 nay bo EMA vi no-it tac dong trong
> che do train head, con ema_study danh gia EMA trong che do fine-tune toan model.

## 4. Bang Ket Qua Chinh

| Run | SHOT F1 | ClipShots F1 | BBC F1 | Ghi chu |
|---|---:|---:|---:|---|
| `A0_autoshot_original` | 0.8405 | 0.7649 | 0.9554 | AutoShot goc, best threshold tung dataset |
| `A1_phase2_bce_onehot` | 0.8378 | 0.6983 | 0.9570 | Control Phase2 toi gian |
| `A2_focal_only` | 0.8378 | 0.6967 | 0.9567 | Focal rieng le |
| `A3_manyhot_only` | 0.8375 | 0.7005 | 0.9563 | Many-hot rieng le |
| `P1_gaussian_only` | 0.8432 | 0.7519 | 0.9436 | Gaussian tren logits A1 |
| `P2_temperature_only` | 0.8378 | 0.6983 | 0.9570 | Temperature tren logits A1 |
| `B1_focal_manyhot` | 0.8384 | 0.7006 | 0.9559 | Focal + many-hot |
| `B4_temperature_gaussian` | 0.8540 | 0.7441 | 0.9570 | Temperature + Gaussian tren logits A1 |
| `B5_full_candidate_no_ema` | 0.8542 | 0.7409 | 0.9551 | Focal + many-hot + temperature + Gaussian |

## 5. Delta So Voi Control A1

| Run | Dataset | Delta F1 | Delta Precision | Delta Recall | Nhan xet |
|---|---|---:|---:|---:|---|
| `A2_focal_only` | SHOT | -0.0000 | -0.0022 | +0.0020 | Gan nhu khong doi |
| `A2_focal_only` | ClipShots | -0.0017 | -0.0023 | -0.0010 | Giam nhe |
| `A2_focal_only` | BBC | -0.0003 | -0.0000 | -0.0006 | Khong dang ke |
| `A3_manyhot_only` | SHOT | -0.0003 | -0.0020 | +0.0012 | Giam rat nhe |
| `A3_manyhot_only` | ClipShots | +0.0021 | +0.0011 | +0.0033 | Cai thien nho |
| `A3_manyhot_only` | BBC | -0.0008 | -0.0004 | -0.0010 | Trong guardrail |
| `P1_gaussian_only` | SHOT | +0.0054 | +0.0185 | -0.0060 | Tang SHOT nhe, precision tang |
| `P1_gaussian_only` | ClipShots | +0.0536 | -0.0111 | +0.1448 | Tang recall manh |
| `P1_gaussian_only` | BBC | -0.0134 | -0.0170 | -0.0101 | Fail guardrail BBC |
| `P2_temperature_only` | SHOT | +0.0000 | +0.0000 | +0.0000 | Khong khac A1 o deploy |
| `P2_temperature_only` | ClipShots | +0.0000 | +0.0000 | +0.0000 | Khong khac A1 o deploy |
| `P2_temperature_only` | BBC | +0.0000 | +0.0000 | +0.0000 | Khong khac A1 o deploy |
| `B1_focal_manyhot` | SHOT | +0.0006 | -0.0040 | +0.0048 | Tang nho |
| `B1_focal_manyhot` | ClipShots | +0.0023 | +0.0013 | +0.0035 | Cai thien nho |
| `B1_focal_manyhot` | BBC | -0.0011 | +0.0002 | -0.0023 | Gan cham guardrail recall |
| `B4_temperature_gaussian` | SHOT | +0.0162 | -0.0001 | +0.0314 | Cai thien ro |
| `B4_temperature_gaussian` | ClipShots | +0.0457 | -0.0228 | +0.1443 | Tang recall manh, precision giam chap nhan duoc |
| `B4_temperature_gaussian` | BBC | -0.0000 | -0.0116 | +0.0105 | F1 gan nhu giu nguyen |
| `B5_full_candidate_no_ema` | SHOT | +0.0164 | -0.0043 | +0.0358 | Cai thien ro |
| `B5_full_candidate_no_ema` | ClipShots | +0.0425 | -0.0283 | +0.1452 | Tang recall manh, precision giam hon B4 |
| `B5_full_candidate_no_ema` | BBC | -0.0019 | -0.0135 | +0.0087 | Trong nguong -0.002 nhung sat guardrail |

## 6. Phan Tich Tung Thanh Phan

### 6.1 AutoShot Goc (`A0_autoshot_original`)

`A0` la checkpoint AutoShot goc, khong train Phase2. Ket qua dung best threshold sweep:

- SHOT F1: 0.8405
- ClipShots F1: 0.7649
- BBC F1: 0.9554

Baseline goc rat manh tren ClipShots khi threshold duoc sweep rieng. Dieu nay cho thay pipeline Phase2 hien tai khong tu dong vuot A0 tren ClipShots neu dung deploy protocol cua ablation.

### 6.2 Control Phase2 (`A1_phase2_bce_onehot`)

`A1` la control chinh: BCE + one-hot, khong EMA, khong many-hot, khong smoothing. Ket qua:

- SHOT F1: 0.8378
- ClipShots F1: 0.6983
- BBC F1: 0.9570

Day la moc de do tac dong cua cac thanh phan train-time va post-process.

### 6.3 Focal Loss Rieng Le (`A2_focal_only`)

Focal Loss khong tao cai thien. F1 gan nhu bang A1 tren SHOT, giam nhe tren ClipShots va BBC. Ket luan: Focal rieng le khong nen duoc trinh bay la thanh phan dong gop chinh.

### 6.4 Many-Hot Rieng Le (`A3_manyhot_only`)

Many-hot tang ClipShots tu 0.6983 len 0.7005, delta +0.0021. Tuy nhien SHOT va BBC giam rat nhe. Ket luan: many-hot co tac dong tich cuc nho tren ClipShots, nhung khong du manh de chon lam candidate rieng.

### 6.5 Gaussian Rieng Le (`P1_gaussian_only`)

Gaussian smoothing tren logits A1 tang ClipShots manh:

- SHOT delta F1: +0.0054
- ClipShots delta F1: +0.0536
- BBC delta F1: -0.0134

Day la bang chung ro rang rang smoothing giup recall va ClipShots, nhung cau hinh Gaussian-only pha guardrail BBC. Khong nen chon `P1` lam deploy.

### 6.6 Temperature Rieng Le (`P2_temperature_only`)

Temperature scaling rieng le khong lam thay doi metric deploy so voi A1. Thanh phan nay chi co y nghia khi ket hop voi Gaussian.

### 6.7 Focal + Many-Hot (`B1_focal_manyhot`)

To hop train-time nay tang nhe SHOT va ClipShots:

- SHOT delta F1: +0.0006
- ClipShots delta F1: +0.0023
- BBC delta F1: -0.0011

Tac dong co nhung nho. Neu chi xet chi phi train lai, `B1` khong dang bang post-process.

### 6.8 Temperature + Gaussian (`B4_temperature_gaussian`)

`B4` la ket qua post-process tot nhat trong matrix:

- SHOT F1: 0.8540, delta +0.0162
- ClipShots F1: 0.7441, delta +0.0457
- BBC F1: 0.9570, delta -0.0000

So voi `P1`, viec them temperature giup giu BBC gan nhu ngang control, trong khi van cai thien lon tren SHOT va ClipShots. Day la cau hinh can duoc uu tien deploy.

### 6.9 Full Candidate Khong EMA (`B5_full_candidate_no_ema`)

`B5` gom Focal + many-hot + temperature + Gaussian, khong EMA:

- SHOT F1: 0.8542, delta +0.0164
- ClipShots F1: 0.7409, delta +0.0425
- BBC F1: 0.9551, delta -0.0019

`B5` cai thien SHOT va ClipShots ro so voi A1, nhung thua `B4` tren ClipShots va BBC. BBC van nam trong guardrail -0.002, nhung sat nguong. Do do `B5` khong phai candidate deploy tot nhat.

## 7. Ket Luan Chon Cau Hinh

Candidate nen chon: `B4_temperature_gaussian`.

Cau hinh deploy de xuat:

```text
source logits/checkpoint: A1_phase2_bce_onehot
temperature: 0.661785970550883
gaussian sigma: 2.0
threshold: 0.1
EMA: off
```

Ly do:

- Tang SHOT F1 tu 0.8378 len 0.8540.
- Tang ClipShots F1 tu 0.6983 len 0.7441.
- BBC gan nhu khong giam: 0.9570 xuong 0.95697.
- Khong can train lai, chi la post-process tren logits/checkpoint A1.
- Tot hon `B5_full_candidate_no_ema` tren ClipShots va BBC.

## 8. Acceptance Criteria

| Tieu chi | Ket qua voi B4 | Trang thai |
|---|---:|---|
| SHOT khong giam qua 0.002 so voi control | +0.0162 | Dat |
| BBC khong giam qua 0.002 so voi control | -0.0000 | Dat |
| ClipShots tang F1 | +0.0457 | Dat |
| Khong dung EMA | EMA off | Dat |
| Khong train lai neu khong can | Chi post-process A1 logits | Dat |

`B5` cung dat nguong BBC voi delta -0.0019, nhung do thua `B4` tren ClipShots va BBC nen khong duoc chon.

## 9. Luu Y Ky Thuat

Trong qua trinh chay `B5`, ket qua ClipShots ban dau ra F1 = 0 do mismatch key: logits dung stem video, trong khi ground truth dung key co duoi `.mp4`. Loi nay da duoc sua bang cach tao ground truth ClipShots theo stem va re-evaluate tu cached logits, khong chay lai 500 video.

Ket qua ClipShots hop le cua `B5` sau khi sua key:

```text
deploy F1 = 0.7408586931
best sweep F1 = 0.7598370050
```

## 10. Huong Tiep Theo

1. Khoa `B4_temperature_gaussian` lam candidate deploy tam thoi.
2. Chay lai `B4` bang workflow inference chinh de xac nhan metric khop voi runner ablation.
3. Neu can cai thien hon A0 tren ClipShots, nen tap trung vao calibration/threshold theo dataset hoac fine-tune co sampling/loss moi. Cac thanh phan train-time hien tai chua vuot duoc A0 tren ClipShots.
4. Viet them bang trong luan van tach ro hai nhom:
   - Baseline goc AutoShot voi best threshold: `A0`.
   - Ablation tren control Phase2: `A1` va cac bien the.

