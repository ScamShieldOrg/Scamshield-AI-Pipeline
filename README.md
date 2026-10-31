# ScamShield-AI-Pipeline

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Architecture](https://img.shields.io/badge/Architecture-2--Tier%20Cascaded%20Router-orange.svg)]()
[![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-blueviolet.svg)](https://git-lfs.com/)

> **Microservice và bộ máy suy luận AI độc lập** chuyên phát hiện tin nhắn lừa đảo tiếng Việt theo thời gian thực trên SMS, OTT, Zalo, Telegram và Facebook, sử dụng kiến trúc định tuyến AI phân tầng hai cấp.

---

## 🏛️ Kiến trúc bộ định tuyến AI hai cấp

```text
SMS → ViSoBERT + tra cứu các URL trong tin nhắn → trả trực tiếp hoặc chuyển hội đồng
URL đứng riêng → PhishTank (feed cục bộ) + Google Safe Browsing → cảnh báo nếu khớp, còn lại UNKNOWN
PHONE / BANK_ACCOUNT → Core backend xử lý; AI Pipeline không nhận hai check_type này
```

Với SMS, yêu cầu có `force_tier2=true` cũng chuyển lên tầng 2 sau khi chạy ViSoBERT.
Tỷ lệ yêu cầu đi mỗi nhánh phụ thuộc vào dữ liệu thực tế; báo cáo trên tập kiểm thử
hiện tại ghi nhận **2,65%** mẫu nằm trong dải REVIEW. Tham số
`TIER1_CONFIDENCE_THRESHOLD` (`τ=0.90`) vẫn được truyền vào router nhưng chưa được
dùng trong điều kiện định tuyến hiện tại.

1. **Cấp 1 (phân loại cục bộ)**:
   - Bộ phân loại **ViSoBERT** đã tinh chỉnh chạy bằng ONNX Runtime trên CPU khi có tệp ONNX hợp lệ; nếu khởi tạo ONNX thất bại, hệ thống thử dùng checkpoint PyTorch trên CUDA hoặc CPU. Tệp ONNX hiện được xuất ở độ chính xác FP32, chưa được lượng tử hóa INT8.
   - Khi huấn luyện, mô hình dùng Cross Entropy có trọng số lớp, với trọng số lớp lừa đảo là **5.0**. Nhiệt độ hiệu chỉnh được học trên tập xác thực; giá trị lưu trong báo cáo huấn luyện hiện tại là **T = 1.4015**. Bộ phân loại áp dụng các ngưỡng 0.35 và 0.85 để đưa ra một trong ba kết quả: HAM, REVIEW hoặc SCAM.
   - URL trong SMS được tra PhishTank và Google Safe Browsing. Nếu URL khớp nguồn đe dọa nhưng ViSoBERT dự đoán HAM, yêu cầu được chuyển tầng 2; khi tầng 2 không khả dụng, kết quả là `UNKNOWN`. URL đứng riêng không chạy ViSoBERT.
   - Không có bộ lọc từ khóa riêng để quyết định Brand Spoofing hay bẫy OTP. Một số heuristics tạo chỉ số radar phục vụ trình bày kết quả; chúng không thay đổi nhãn phân loại của ViSoBERT.
2. **Cấp 2 (hội đồng phân tích chuyên sâu)**:
   - Với SMS, chạy khi ViSoBERT trả `REVIEW`, tín hiệu URL mâu thuẫn với kết quả HAM, hoặc yêu cầu đặt `force_tier2=true`.
   - Hội đồng gồm ba vai trò cấu hình trong `config/council_registry.yaml`:
     1. `Cyber-Financial & Banking Security Analyst`: URL, hạ tầng kỹ thuật và quy trình ngân hàng.
     2. `Psycholinguistic & Vietnamese Slang Profiler`: thao túng tâm lý, tiếng lóng và teencode.
     3. `Public Defender & Anti-False-Positive Advocate`: tìm căn cứ hợp lệ để hạn chế báo động giả.
   - Trước khi gửi đến hội đồng, hệ thống che số điện thoại, giấy tờ tùy thân và số tài khoản phát hiện được. Các vai trò chạy song song; mỗi vai trò có mô hình phân tích chính (*maker*), mô hình thẩm định (*checker*) và chuỗi mô hình dự phòng. Nếu các lời gọi đám mây thất bại, vai trò dùng heuristic cục bộ.
   - Khi RAG khả dụng, đầu vào được chuẩn hóa và che PII một lần, rồi BGE-M3 truy xuất tối đa 5 đoạn đã kiểm duyệt vượt ngưỡng cosine được hiệu chỉnh. Hội đồng và nhánh Gemini dự phòng dùng cùng chứng cứ có `chunk_id`; similarity không phải xác suất lừa đảo.
   - Điểm đồng thuận là trung bình xác suất lừa đảo theo trọng số vai trò. Khi Defender bỏ phiếu `LOW-RISK` với độ tin cậy ít nhất 0.80, điểm giảm 0.25. Hội đồng trả `SCAM-RISK`, `LOW-RISK` hoặc `UNKNOWN` theo ngưỡng và tình trạng mâu thuẫn phiếu. Cấu hình Early-Exit hiện yêu cầu **3 phiếu đồng nhất trên 3 vai trò**, nên không bỏ qua vai trò nào trong cấu hình mặc định. Circuit Breaker tạm ngắt mô hình lỗi và thử mô hình dự phòng.
   - Nếu hội đồng gặp lỗi ở cấp điều phối, router thử một Gemini LLM qua `ModelRegistry`; nếu tầng 2 vẫn không khả dụng, router giữ kết quả ViSoBERT, bao gồm nhãn `REVIEW` khi chưa thể kết luận. Trường hợp URL khớp nguồn đe dọa nhưng ViSoBERT trả HAM giữ `UNKNOWN`. Nhánh Gemini LLM đơn lẻ hiện chưa dùng bước che PII của hội đồng.

---

## 📁 Cấu trúc Thư mục

```text
Scamshield-AI-Pipeline/
├── .env.example              # Mẫu biến môi trường API keys
├── .gitattributes            # Cấu hình Git LFS (track *.onnx, *.safetensors)
├── .gitignore                # Bỏ qua bytecode, venv, secrets
├── Dockerfile                # Build container độc lập
├── docker-compose.yml        # Chạy microservice với 1 lệnh
├── requirements.txt          # Danh sách thư viện Python
├── README.md                 # Tài liệu hướng dẫn
├── demo_scamshield_mvp.py    # Demo tương tác dòng lệnh
├── run_demo.bat              # Shortcut chạy demo trên Windows
│
├── api/                      # REST API phục vụ Backend & App
│   ├── app.py                # FastAPI endpoints (/api/v1/scan, /health)
│   └── schemas.py            # Pydantic data contracts (chuẩn hóa I/O)
│
├── config/                   # Cấu hình độc lập (Zero-hardcoding)
│   ├── council_registry.yaml # Cấu hình 3 vai trò Hội đồng Pháp y
│   └── gemini_models.yaml    # Cấu hình Dynamic Model Registry
│
├── models/                   # Trọng số & Tokenizer Tier 1 (Quản lý qua Git LFS)
│   ├── visobert_finetuned/   # Checkpoint, tokenizer, ONNX, training_report.json
│   └── scamshield_config.json # Cấu hình cũ; bộ suy luận đọc training_report.json
│
├── pipeline/                 # Core AI Engine
│   ├── two_tier_router.py    # Orchestrator định tuyến 2 tầng
│   ├── visobert_classifier.py # ViSoBERT: ưu tiên ONNX CPU, dự phòng PyTorch
│   ├── forensic_council.py   # Hội đồng 3 vai trò (Maker-Checker, Defender)
│   ├── pii_masking.py        # Che PII trước lời gọi hội đồng
│   ├── persistence.py        # Lưu kết quả vào PostgreSQL
│   └── model_registry.py     # Dynamic Cloud Model Registry
│
├── scripts/                  # Scripts benchmark, training & ONNX export
│   ├── train_visobert.py     # Tinh chỉnh ViSoBERT với trọng số lớp SCAM = 5.0
│   ├── export_visobert_onnx.py # Xuất ONNX opset 17 (FP32)
│   └── evaluate_council_benchmark.py
│
└── tests/                    # Kiểm thử tự động
    └── test_api.py
```

---

## 🚀 Hướng dẫn cài đặt và chạy

### 1. Chuẩn bị môi trường

```bash
# Sao chép kho mã nguồn
git clone https://github.com/thalord2825/Scamshield-AI-Pipeline.git
cd Scamshield-AI-Pipeline

# Kích hoạt Git LFS để tải các tệp mô hình lớn (*.onnx)
git lfs install
git lfs pull
```

### 2. Cài đặt thư viện Python

```bash
# Khuyến nghị dùng Python 3.10–3.12
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Cấu hình biến môi trường

Tạo tệp `.env` từ mẫu `.env.example`:
```bash
cp .env.example .env
```
Điền `OPENAI_API_KEY` hoặc `ANTHROPIC_API_KEY` cho mô hình maker qua proxy
`VIBI_BASE_URL`, và `GEMINI_API_KEY` hoặc `GOOGLE_API_KEY` cho checker, mô hình dự
phòng và nhánh Gemini LLM đơn lẻ. Nếu không có khóa dùng được, từng vai trò của
hội đồng có thể trả kết quả heuristic cục bộ; đó không phải kết quả từ LLM.

### Dữ liệu URL cho Tier 1

`SAFE_BROWSING_API_KEY` là khóa riêng cho Safe Browsing v5, không dùng chung biến
`GOOGLE_API_KEY` của Gemini. Bộ tra cứu giải mã phản hồi Protocol Buffers của
`urls.search` và dùng `cacheDuration` do Google trả về. Khi API khởi động, nếu
chưa có feed PhishTank ở
`data/phishtank_online_valid.csv.gz`, nó tải feed công khai một lần trong nền.
Nếu đã có file, Tier 1 dùng dữ liệu đó ngay, kể cả khi file đã cũ; không tải
lại định kỳ. Có thể để trống `PHISHTANK_APP_KEY`; nếu có khóa thì lần tải dùng
khóa đó. PhishTank giới hạn lượt tải feed công khai không có khóa. File được lưu
trong thư mục `data/` của project (Compose cũng dùng thư mục này) và không được
commit. Lần tra cứu đầu tiên có thể thấy PhishTank `UNAVAILABLE` trong lúc tải.
Nếu tải thất bại, lần khởi động API tiếp theo sẽ thử lại. Có thể chạy
`python scripts/update_phishtank.py` để tải thủ công khi thiếu file, hoặc
`python scripts/update_phishtank.py --force` để chủ động thay bản hiện có.
Đặt `PHISHTANK_AUTO_DOWNLOAD=false` nếu không muốn API tự tải.

Feed cũ là ảnh chụp tại thời điểm tải; một URL từng nằm trong danh sách có thể
đã thay đổi trạng thái. Cần xem thời điểm file trong telemetry khi đánh giá kết
quả. URL không khớp nguồn nào được trả `UNKNOWN`, không tự kết luận `SAFE`.
Trong signal Safe Browsing, `lookup_status=SUCCESS` và `http_status=200` nghĩa là
đã tra cứu thành công; `known_threat_match=false` cùng `status=NO_MATCH` chỉ
nghĩa là không khớp danh sách đe dọa đã biết. Khi chưa chạy hoặc tra cứu lỗi,
`known_threat_match=null` và `lookup_status` lần lượt là `NOT_RUN` hoặc `FAILED`.
Nguồn Safe Browsing chỉ được phép dùng phi thương mại; sản phẩm thương mại cần
thay bằng Google Web Risk. `urls.search` gửi URL gốc cho Google, gồm cả đường
dẫn và query; cân nhắc dữ liệu nhạy cảm trước khi bật khóa này.

Tier 1 chỉ so khớp URL, không tải trang hay theo chuyển hướng. `confidence=1.0`
cho URL khớp là chỉ báo **khớp danh sách nhị phân**, không phải xác suất mô hình;
`telemetry.tier1.confidence_kind` ghi rõ điều này.
Khi kết quả là `UNKNOWN`, `risk_score` là `null` để không hiển thị nhầm thành
điểm rủi ro thấp.
Đánh giá trên CSV độc lập có cột `url,label` (`1` là URL độc hại đã xác minh,
`0` là URL lành tính đã xác minh) bằng
`python scripts/evaluate_url_reputation.py path/to/url_test.csv`. Báo cáo tách
tỷ lệ cảnh báo sai, tỷ lệ phát hiện và tỷ lệ `UNKNOWN`; không tính `UNKNOWN` là
`SAFE`. Repo hiện chưa có tập URL độc lập như vậy.

### 4. Khởi chạy dịch vụ REST API

Endpoint quét lưu kết quả hoàn tất vào PostgreSQL. Khi chạy cục bộ, hãy khởi động
các dịch vụ bằng Compose để có sẵn cơ sở dữ liệu, hoặc đặt `DATABASE_URL` trỏ đến
một máy chủ PostgreSQL có thể truy cập. Hai endpoint `POST /api/v1/scan` và
`POST /api/v1/visobert/predict` nhận tùy chọn `core_check_id` là số nguyên dương.
Truyền `id` tương ứng trong bảng `check_requests` của Core để các lần thử lại cập
nhật cùng một tác vụ. Nếu không truyền, dịch vụ tự tạo mã tham chiếu nội bộ và
ghi mỗi yêu cầu thành một tác vụ riêng. Hãy truyền mã Core khi cần loại bỏ các
tác vụ trùng lặp do thử lại.

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
Tài liệu API được tạo tự động tại:
- Swagger UI: `http://localhost:8000/docs`
- Redoc: `http://localhost:8000/redoc`

### 5. Chạy bằng Docker hoặc Docker Compose

```bash
# Khởi động các dịch vụ bằng Docker Compose
docker compose up -d --build

# Theo dõi nhật ký hoạt động
docker compose logs -f
```

Compose khởi động PostgreSQL có bật pgvector trước dịch vụ AI. Trong lần khởi tạo
cơ sở dữ liệu đầu tiên, Postgres chạy `db/init/001_ai_service.sql`, tệp DDL tương
ứng với `ScamShield_App/docs/database/ai-service.dbml`. Lược đồ tạo các bảng tác
vụ AI, mô hình, suy luận và RAG. Các tập lệnh khởi tạo chỉ chạy khi volume dữ liệu
Postgres còn trống. Nếu volume đã có cơ sở dữ liệu, hãy áp dụng SQL một lần bằng
lệnh `psql -f db/init/001_ai_service.sql`.

### RAG cho Tier 2

Khi `RAG_ENABLED=true` (mặc định), API tải **một** BGE-M3 tại startup và dùng chung
cho suy luận và tác vụ nạp nguồn chạy nền. Model được ghim tại revision
`5617a9f61b028005a4858fdac845db406aefb181`; lần đầu cần tải từ Hugging Face.
Compose lưu cache trong volume `bge-m3-model-cache` và chạy một Uvicorn worker.
Không chạy nhiều worker nếu muốn chỉ có một bản model trong tiến trình dịch vụ.
Nếu tải model hoặc DB lỗi, Tier 2 tiếp tục chạy và báo `telemetry.tier2.rag.status=UNAVAILABLE`.
Trước Tier 2, SMS được chuẩn hóa và che PII; URL trong prompt chỉ giữ scheme,
hostname và dấu hiệu liên kết APK, còn Tier 1 vẫn tra URL gốc.

Tác vụ nền đọc URL hạt giống trong `config/rag_sources.yaml` mỗi 24 giờ. Chỉ URL
HTTPS có hostname nằm chính xác trong `allowed_hosts` được tải; redirect bị từ
chối. Các trang đã đọc được nạp thành `VERIFIED` và có hiệu lực 7 ngày sau lần
tải thành công gần nhất. Có thể đặt ca đã duyệt tại
`data/rag/approved_cases/*.json` với các trường bắt buộc `title`, `text`,
`review_status: "VERIFIED"`, `case_label: "SCAM"` hoặc `"HAM"`; `source_url` là
tùy chọn. Nội dung được che PII trước khi lưu và tạo embedding. Khi nguồn đổi,
phiên bản cũ ngừng hoạt động nhưng vẫn còn cho lịch sử trích dẫn.

**RAG không tự kích hoạt chỉ vì có corpus.** Tạo benchmark tiếng Việt JSONL với
`query_id`, `query`, `passage`, `relevant` (boolean), `source_group` và `split`
(`calibration` hoặc `holdout`). Mỗi `source_group` chỉ thuộc một split; các cặp
query–passage cần được người đánh giá gán nhãn liên quan. Chạy:

```bash
python scripts/calibrate_rag.py data/rag/relevance_benchmark.jsonl
```

Script chỉ ghi `config/rag_threshold.json` khi precision của các đoạn được nhận
đạt ít nhất 95% ở cả calibration và holdout. Đưa file này vào image khi deploy.
Nếu file thiếu, sai revision hoặc chưa đạt kiểm định, RAG báo `UNAVAILABLE` và
không đưa Top-5 yếu vào prompt. Khi có ngưỡng, mỗi lần quét chọn Top-5 cosine
trên tài liệu `VERIFIED`, còn hiệu lực và đúng embedding model/version, rồi loại
mọi đoạn dưới ngưỡng; nếu không còn đoạn nào, trả `NO_MATCH`. Similarity chỉ là
độ gần trong không gian embedding, không phải xác suất lừa đảo.

Repo có benchmark khởi đầu `data/rag/relevance_benchmark.jsonl` gồm 40 cặp đã
gán nhãn theo bốn nhóm bài viết; script giải tham chiếu `source_key` và `chunk_no`
từ corpus active. Revision hiện tại tạo ngưỡng `0.6536799` trong
`config/rag_threshold.json`: precision 100% ở cả hai split, recall lần lượt
37,5% và 25%. Đây là mẫu nhỏ, ưu tiên tránh chứng cứ sai; cần mở rộng tập
gán nhãn trước khi coi các số này là chất lượng triển khai thực tế.

API tự áp dụng `db/migrations/002_rag.sql` cho volume PostgreSQL đã tồn tại trước
khi khởi tạo RAG. Để áp dụng thủ công: `psql "$DATABASE_URL" -f db/migrations/002_rag.sql`.

So sánh kết quả SMS trước và sau RAG trên cùng tập kiểm thử bằng hai lần chạy
`scripts/run_two_tier_benchmark.py` với `--mode cascaded --dataset
data/raw/vietnamese_sms_test.csv`; thêm `--rag` ở lần thứ hai. Script báo
accuracy, FPR, recall SCAM, độ trễ và chi phí, đồng thời lưu telemetry từng mẫu.
Khi có `--rag`, script không dùng cache lời gọi hội đồng của lần baseline.

Trước khi triển khai ngoài môi trường phát triển cục bộ, hãy đặt `POSTGRES_PASSWORD` trong tệp `.env`.

---

## 📡 Các endpoint API

### 1. Kiểm tra trạng thái hệ thống: `GET /health`
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "service": "ScamShield AI Pipeline Engine",
  "tier1_ready": true,
  "tier2_ready": true,
  "database_ready": true
}
```

### 2. Quét SMS hoặc URL: `POST /api/v1/scan`

`content` là trường bắt buộc chứa nội dung cần quét. Endpoint này không còn nhận
`message` làm tên thay thế. `check_type` là trường bắt buộc, chỉ nhận `SMS` hoặc
`URL`; Core xử lý `PHONE` và `BANK_ACCOUNT` mà không gọi AI Pipeline.

**Yêu cầu:**
```json
{
  "content": "Vietcombank: Tai khoan cua ban tam khoa do bat thuong. Dang nhap https://vcb-xacthuc.xyz de mo khoa.",
"check_type": "SMS"
}
```

**Phản hồi minh họa** (định dạng chuẩn hóa cho Spring Boot và giao diện người dùng;
`scan_id`, thời gian, xác suất và độ trễ thay đổi theo từng lần chạy):
```json
{
  "status": "success",
  "data": {
    "scan_id": "SC-20261018-140FCE",
    "timestamp": "2026-10-18T14:17:23.126590+00:00",
    "tier_executed": 1,
    "verdict": {
      "risk_level": "DANGEROUS",
      "risk_score": 99.6,
      "scam_category": "ViSoBERT text classifier",
      "confidence": 0.996,
      "summary": "ViSoBERT classified this message as SCAM with calibrated scam probability 0.9960."
    },
    "radar_metrics": {
      "urgency": 15,
      "authority_impersonation": 75,
      "financial_lure": 15,
      "suspicious_channel": 70,
      "data_harvesting": 10
    },
    "council_breakdown": null,
    "actionable_recommendations": [
      "Tuyệt đối không nhấp vào bất kỳ đường link nào trong tin nhắn.",
      "Không cung cấp mật khẩu ngân hàng, mã OTP, số CCCD hoặc thông tin sinh trắc học.",
      "Thông báo cho người thân, chặn số liên lạc và gửi báo cáo lừa đảo lên hệ thống."
    ],
    "telemetry": {
      "routing_path": "TIER_1_DIRECT_FASTPATH",
      "routing_reason": "ViSoBERT P(SCAM)=0.9960 falls outside REVIEW band [0.35, 0.85].",
      "total_latency_ms": 651.79,
      "total_cost_usd": 0,
      "tier1": {
        "label": "SCAM",
        "class_id": 1,
        "confidence": 0.996,
        "scam_prob": 0.996,
        "threshold_low": 0.35,
        "threshold_high": 0.85,
        "temperature": 1.4015225172042847,
        "engine": "onnx",
        "model_version": "1",
        "ham_prob": 0.004,
        "latency_ms": 651.26,
        "cost_usd": 0,
        "is_sure": true,
        "status": "CONFIDENT",
        "matched_cues": [],
        "reason": "ViSoBERT P(SCAM)=0.9960; REVIEW band=[0.35, 0.85]."
      },
      "tier2": {
        "status": "SKIPPED",
        "reason": "FastPath satisfied; no cloud escalation needed."
      },
      "final_label": "SCAM"
    }
  },
  "error": null
}
```

---

## 🧪 Huấn luyện, chạy demo và kiểm thử

### Tinh chỉnh ViSoBERT cho Cấp 1

Tệp dữ liệu huấn luyện là `data/raw/vietnamese_sms_merged_all.csv` (gồm các cột
`message,label`; có thể có thêm `message_id,source`). Tệp kiểm thử độc lập là
`data/raw/vietnamese_sms_test.csv` (gồm `message,label`). Nhãn `0=HAM` biểu thị
tin nhắn bình thường, `1=SCAM` biểu thị tin nhắn lừa đảo. Tập kiểm thử được cung
cấp có một số nội dung trùng với tệp huấn luyện; trước khi tạo tập xác thực phân
tầng, `scripts/train_visobert.py` loại các tin nhắn kiểm thử khỏi dữ liệu huấn
luyện. Quá trình sẽ dừng nếu phát hiện nhãn mâu thuẫn.

```powershell
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe scripts/train_visobert.py
```

Trong lần chạy đầu tiên, tập lệnh tải checkpoint chính thức `uitnlp/visobert` theo
phiên bản `model.safetensors` đã ghim. Trên Windows có trình điều khiển NVIDIA
tương thích, hãy cài bản PyTorch hỗ trợ CUDA từ kho chính thức của PyTorch trước
khi huấn luyện; gói pip mặc định có thể chỉ hỗ trợ CPU. Tệp kiểm thử có 597 dòng
nhưng chỉ gồm 566 tin nhắn khác nhau, vì vậy số mẫu kiểm thử được báo cáo là 566.

Tập lệnh tinh chỉnh đầu phân loại hai lớp với trọng số lớp lừa đảo bằng 5, chọn
checkpoint có macro-F1 trên tập xác thực cao nhất, hiệu chỉnh nhiệt độ từ logits
xác thực rồi đánh giá một lần trên tập kiểm thử độc lập. Các tệp mô hình,
tokenizer và `training_report.json` được lưu trong `models/visobert_finetuned/`.
Dùng các tùy chọn `--epochs`, `--batch-size`, `--max-length` và `--output-dir` để
điều chỉnh lần chạy. Checkpoint `.safetensors` được quản lý bằng Git LFS theo cấu
hình của kho mã nguồn.

`models/visobert_finetuned/training_report.json` ghi nhận 1.781 mẫu train,
314 mẫu validation và 566 mẫu test sau khi loại trùng. Trên tập test, accuracy
đạt **90,64%**, macro-F1 **88,16%**, recall lớp SCAM **94,78%** và tỷ lệ REVIEW
**2,65%**. Các số liệu này thuộc tập kiểm thử được cung cấp, không phải ước lượng
tỷ lệ lưu lượng hay chất lượng trên dữ liệu triển khai thực tế.

### Xuất bộ phân loại đã huấn luyện sang ONNX

```powershell
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe scripts/export_visobert_onnx.py
```

Lệnh này tạo `models/visobert_finetuned/visobert_classifier.onnx` và
`onnx_export_report.json`. Công cụ xuất kiểm tra đồ thị ONNX và so sánh logits
trên CPU của ONNX Runtime với PyTorch bằng một số tin nhắn mẫu. Thông tin hiệu
chỉnh và ngưỡng quyết định vẫn nằm trong `training_report.json`. Bộ suy luận của
API ưu tiên tệp ONNX hợp lệ qua ONNX Runtime trên CPU; nếu khởi tạo thất bại, nó
thử dùng checkpoint PyTorch trong cùng thư mục.

- **Chạy bản demo dòng lệnh với 10 trường hợp kiểm thử chuẩn:**
  ```bash
  python demo_scamshield_mvp.py
  ```
- **Chạy demo tương tác để nhập tin nhắn trực tiếp:**
  ```bash
  python demo_scamshield_mvp.py --interactive
  ```

---

## 🛡️ Giấy phép và đội ngũ
Phát triển bởi đội ngũ ScamShield-VN Core AI Engineering Team.
Mọi đóng góp cần tuân thủ quy chuẩn mã nguồn mở và bảo đảm an toàn thông tin, bảo mật dữ liệu.
