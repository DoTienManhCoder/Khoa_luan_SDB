# AutoShotV2 Web

Ứng dụng web nằm trong cùng monorepo với package huấn luyện và suy luận:

- `apps/web/frontend`: React, TypeScript và Vite.
- `apps/web/backend`: FastAPI, MongoDB, Cloudinary hoặc lưu trữ local.
- `src/autoshotv2/runtime.py`: runtime model dùng chung cho CLI và backend.
- `artifacts/models/`: checkpoint triển khai cục bộ, không được đưa vào Git.

## Chạy Bằng Docker

Đặt checkpoint tại `artifacts/models/deploy.pth`. Nếu chưa có checkpoint, backend `auto`
vẫn chạy bằng OpenCV baseline.

CPU:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

GPU NVIDIA:

```powershell
Copy-Item .env.example .env
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Mở frontend tại `http://localhost:5173` và health API tại
`http://localhost:8000/api/health`.

## Chạy Khi Phát Triển

MongoDB phải chạy trước. Từ thư mục gốc:

```powershell
pip install -r requirements.txt
pip install -r apps/web/backend/requirements.txt
pip install -e .

Set-Location apps/web/backend
uvicorn app.main:app --reload
```

Trong terminal khác:

```powershell
Set-Location apps/web/frontend
npm ci
npm run dev
```

Khi chạy backend ngoài Docker, đặt `STORAGE_DIR` và
`AUTOSHOT_MODEL_PATH` thành đường dẫn phù hợp trên máy.

## Chọn Backend

Form upload hỗ trợ ba chế độ:

| Giá trị | Hành vi |
|---|---|
| `auto` | Dùng Phase2 nếu checkpoint hợp lệ, nếu không dùng OpenCV baseline |
| `phase2` | Bắt buộc dùng AutoShotV2; job lỗi nếu model không khả dụng |
| `baseline` | Luôn dùng OpenCV histogram trên CPU |

Tham số hậu xử lý có thứ tự ưu tiên:

1. Giá trị gửi theo từng job.
2. `AUTOSHOT_DEFAULT_TEMPERATURE`, `AUTOSHOT_DEFAULT_SIGMA`,
   `AUTOSHOT_DEFAULT_THRESHOLD`.
3. Metadata `phase2_config` trong checkpoint.
4. Mặc định deploy: temperature `0.38780461844336944`, sigma `2.0`,
   threshold `0.1`.

## API

| Method | Endpoint | Chức năng |
|---|---|---|
| `GET` | `/api/health` | Trạng thái MongoDB, checkpoint và device |
| `POST` | `/api/jobs/from-upload` | Upload video và tạo job |
| `GET` | `/api/jobs/{job_id}` | Trạng thái và kết quả job |
| `GET` | `/api/jobs/{job_id}/scenes` | Danh sách scene |
| `GET` | `/api/jobs/{job_id}/boundaries` | Danh sách boundary |
| `GET` | `/api/jobs/{job_id}/exports/{kind}` | Tải `json`, `csv` hoặc `txt` |
| `DELETE` | `/api/jobs/{job_id}` | Xóa metadata và artifact của job |

MongoDB tạo TTL index trên `expires_at`. Cloudinary chỉ được bật khi đủ ba
credential; nếu không, file được phục vụ từ `/media`.
