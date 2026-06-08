# Triển Khai AutoShotV2 Web

Không commit `.env`, checkpoint hoặc credential. Dùng `.env.example` làm mẫu.

## Biến Môi Trường Backend

```text
MONGODB_URI=mongodb+srv://...
MONGODB_DB=autoshot
STORAGE_DIR=/app/storage
ALLOWED_ORIGINS=https://your-frontend.example
MAX_UPLOAD_MB=300
MAX_DURATION_SEC=600
JOB_TTL_HOURS=24

CLOUDINARY_CLOUD_NAME=...
CLOUDINARY_API_KEY=...
CLOUDINARY_API_SECRET=...
CLOUDINARY_FOLDER=autoshot

AUTOSHOT_MODEL_PATH=/app/models/deploy.pth
AUTOSHOT_DEVICE=auto
AUTOSHOT_USE_BASELINE=false
```

Ba biến `AUTOSHOT_DEFAULT_*` là tùy chọn. Để trống để dùng metadata checkpoint
và giá trị deploy chuẩn.

Backend cần:

- Python 3.10 trở lên.
- `ffmpeg` trong `PATH`.
- MongoDB truy cập được.
- Volume hoặc image chứa checkpoint tại `AUTOSHOT_MODEL_PATH`.
- Persistent storage nếu không dùng Cloudinary.

CPU dùng `apps/web/backend/Dockerfile`. GPU NVIDIA dùng
`apps/web/backend/Dockerfile.gpu` và Docker runtime hỗ trợ GPU.

## Biến Môi Trường Frontend

```text
VITE_API_BASE_URL=https://your-api.example
```

`VITE_API_BASE_URL` được đóng vào bundle lúc build. Host static cần rewrite mọi
route frontend về `index.html`.

## Tách Frontend Và Backend

Có thể deploy frontend lên Vercel/Netlify và backend lên dịch vụ container.
Khi đó:

1. Đặt `VITE_API_BASE_URL` thành URL backend.
2. Đặt `ALLOWED_ORIGINS` thành origin chính xác của frontend.
3. Dùng MongoDB Atlas hoặc MongoDB quản lý riêng.
4. Dùng Cloudinary hoặc volume bền vững cho upload và export.
5. Mount checkpoint ngoài Git vào container backend.

Endpoint `/api/health` cho biết MongoDB có hoạt động, checkpoint có tồn tại và
backend nào sẽ được ưu tiên.
