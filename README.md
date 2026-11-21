# Castle5000: dataset manifest

Pipeline hiện có: quét ảnh, tạo manifest, VLM caption, tìm kiếm BM25 trên caption
và dense retrieval ảnh bằng SigLIP 2 Large. Chưa có giao diện hay nộp bài DRES.

## Cài đặt và chạy

```powershell
python -m pip install -r requirements.txt
python src/dataset.py
```

## Dữ liệu

```text
data/
  Batch1/*.jpg
  Batch2/Batch2/*.jpg
  Batch3/*.jpg
```

Chuong trinh quet de quy, nen van doc duoc Batch2 long hai cap. Khong doi ten,
di chuyen hoac chinh sua anh goc.

## Chay

Can Python 3.11+ va Pillow (de kiem tra cau truc va giai ma toan bo anh).

```powershell
python -m pip install -r requirements.txt
python src/dataset.py
```

Mac dinh doc thu muc `data` va ghi `artifacts/manifest.csv` tuong doi voi vi tri
du an, khong phu thuoc thu muc dang chay lenh. Cac duong dan tuy chinh duoc tinh
theo thu muc dang chay lenh:

```powershell
python src/dataset.py --data-root data --output artifacts/manifest.csv --expected-count 5000
```

Mặc định bắt buộc đủ 5.000 ảnh. Để thử với tập con, truyền số ảnh mong đợi qua
`--expected-count N`, hoặc `--expected-count 0` để tắt kiểm tra số lượng.
Thư mục rỗng vẫn bị từ chối. Chỉ quét `.jpg`, `.jpeg`, `.png` (không phân biệt
hoa thường); các file khác bị bỏ qua.

## Schema CSV

| Cột | Ý nghĩa |
| --- | --- |
| `imageID` | ID để nộp DRES: `day1_Kitchen_` + tên file bỏ đuôi; giữ nguyên số 0 đầu |
| `file_path` | Đường dẫn tương đối với **data-root**, dùng dấu `/` |
| `collection` | Luôn là `castle5000` |
| `batch` | Thư mục đầu tiên bên dưới data-root, ví dụ `Batch2` |
| `hour` | Phần HH trong tên file, số nguyên 0-23 |
| `frame_index` | Phần frame trong tên file, số nguyên; không phải timestamp |
| `width`, `height` | Kích thước pixel đọc từ ảnh |
| `size_bytes` | Dung lượng file |
| `sha256` | Hash nội dung file để đối chiếu dữ liệu |

Ví dụ: `data/Batch2/Batch2/11_2625.jpg` có `file_path` là
`Batch2/Batch2/11_2625.jpg`, `imageID` là `day1_Kitchen_11_2625`.
Đây là manifest tự tạo theo quy tắc tên trong PDF, không phải metadata chính
thức của BTC; không suy diễn nhãn ngữ cảnh từ nội dung ảnh.

Mỗi đường dẫn ảnh được khôi phục bằng `Path(data_root) / row["file_path"]`.
Thứ tự dòng ổn định theo đường dẫn. Không dùng số thứ tự dòng làm ID nộp bài.

## Kiểm tra và lỗi

- Từ chối tên không đúng `HH_frame`, giờ ngoài 00-23, ID trùng giữa các batch,
  ảnh hỏng/không giải mã được, symlink ảnh, và số lượng không khớp.
- Nếu có lỗi, thoát với mã 1 và không ghi đè manifest cũ. Chỉ hiển thị tối đa
  20 chi tiết lỗi. Manifest hợp lệ được thay thế bằng thao tác atomic.
- File có nội dung trùng nhau nhưng khác ID vẫn được giữ; số bản trùng được
  báo ở console. Không tự động xóa ảnh vì các frame khác nhau có thể giống nhau.
- Không thay đổi dữ liệu trong lúc đang quét.

Chạy kiểm tra:

```powershell
python -m unittest discover -s tests -v
```

## VLM caption và BM25

Chạy các lệnh sau từ thư mục gốc dự án (dùng `python -m src...`). Mô hình mặc
định là [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct),
chạy local, không cần API key. Lần đầu sẽ tải trọng số nhiều GB vào
`artifacts/hf_cache/`; cần Internet, dung lượng đĩa và RAM/VRAM phù hợp.

```powershell
python -m pip install -r requirements-caption.txt
# Thử 20 ảnh mới trước, đọc caption để kiểm tra chất lượng:
python -m src.caption --device cuda --limit 20
# Chỉ mục thử nghiệm, cho phép chưa có caption toàn bộ manifest:
python -m src.bm25 build --allow-partial
python -m src.bm25 search "người mặc áo đỏ cạnh tủ lạnh" --top-k 10
# Tiếp tục các ảnh còn thiếu, sau khi đã kiểm tra caption:
python -m src.caption --device cuda
# Bản đầy đủ: bắt buộc mọi ảnh trong manifest đều có caption khớp hash/đường dẫn:
python -m src.bm25 build
```

`--device auto` (mặc định) chọn CUDA nếu PyTorch hỗ trợ; ngược lại dùng CPU.
Máy có GPU NVIDIA nhưng bản PyTorch CPU vẫn không dùng được GPU. Kiểm tra bằng
`python -c "import torch; print(torch.cuda.is_available())"`; cài bản PyTorch
phù hợp theo [hướng dẫn chính thức](https://pytorch.org/get-started/locally/).
CPU chạy được nhưng chậm và cần nhiều RAM; không nên chạy 5.000 ảnh bằng CPU
trước khi đo thời gian trên tập nhỏ. Chưa có cam kết về batch size/VRAM cho mọi máy.

### Cách lưu và tiếp tục

- `artifacts/lexical.sqlite` chứa bảng `captions`, `metadata` và chỉ mục
  `caption_fts`. Không cần dịch vụ database, không dùng pickle.
- Mỗi ảnh có `imageID`, `file_path`, `sha256`, `caption`; giữ quan hệ với manifest.
  Caption lưu ngay sau từng ảnh. Chạy lại sẽ bỏ qua các ảnh đã hoàn thành.
- Chỉ tái sử dụng caption cho file có SHA-256 giống hệt; không tái sử dụng cho
  ảnh chỉ gần giống nhau. Mỗi `imageID` vẫn có bản ghi riêng.
- Prompt, model, revision, giới hạn pixel/token được lưu trong metadata. Nếu đổi
  model, prompt hay cấu hình tiền xử lý, dùng `--db artifacts/lexical_v2.sqlite`
  để tránh trộn caption. Riêng `--max-new-tokens` được phép đổi trên cùng database;
  caption hoàn chỉnh đã lưu được giữ lại, cấu hình cũ lưu trong metadata history.
  Có thể truyền `--revision <commit-hash>` để ghim phiên bản mô hình; mặc định
  `main` có thể thay đổi theo thời gian. Prompt lưu trực tiếp trong `src/caption.py`.
- Thêm/sửa caption làm mất hiệu lực index cũ; chạy lại `bm25 build`.
  Build chỉ mục diễn ra trong transaction, lỗi sẽ rollback.
- `--limit` là số caption **mới** trong lần chạy, không phải tổng số ảnh đích.
  Chỉ chạy một tiến trình ghi caption/index cho mỗi database tại một thời điểm.
- `--manifest`, `--data-root`, `--db`, `--cache-dir` cho phép đổi vị trí.

Xem caption đã tạo:

```powershell
python -c "import sqlite3; c=sqlite3.connect('artifacts/lexical.sqlite'); print(*c.execute('SELECT imageID, caption FROM captions LIMIT 5'), sep='\n')"
```

### Thiết kế và giới hạn

Để đổi **chỉ prompt** và tiếp tục dùng database hiện có:

```powershell
python -m src.caption --device cuda --db artifacts/lexical.sqlite --allow-prompt-change
```

Caption đã có được giữ nguyên; prompt mới chỉ dùng cho ảnh chưa hoàn thành.
Cấu hình trước đó được lưu trong `caption_config_history`. Sau khi chấp nhận
prompt mới, lần chạy tiếp theo không cần flag nếu prompt không đổi nữa. Flag
không cho phép đổi model, ngôn ngữ hoặc tiền xử lý. Database sẽ chứa caption
từ nhiều phiên bản prompt; cache theo hash bị tắt khi lịch sử có prompt khác
để ảnh mới không nhận nhầm caption từ prompt cũ. Build lại BM25 sau khi bổ sung
caption bằng `python -m src.bm25 build --db artifacts/lexical.sqlite`.

- Một caption tiếng Việt ngắn cho mỗi ảnh, ưu tiên người, trang phục, đồ vật,
  tư thế, vị trí. Đây là dự đoán từ một ảnh tĩnh, không phải ground truth QA.
  Chưa tạo tags/OCR riêng để giữ baseline đơn giản.
- Inference từng ảnh, nạp mô hình một lần, `inference_mode`, không sampling;
  giới hạn pixel để kiểm soát bộ nhớ. Ngân sách mặc định 768 token; nếu chạm
  giới hạn, tự thử lại một lần với ngân sách gấp đôi. Nếu vẫn chưa hoàn chỉnh,
  báo lỗi thay vì lưu văn bản bị cắt. Có thể tăng `--max-new-tokens` và chạy lại
  trên cùng database, không cần xóa caption đã lưu.
  Khi cả hai lần đều chạm giới hạn, văn bản đầu ra nguyên gốc (giữ token đặc biệt),
  token IDs, số token, prompt và đường dẫn ảnh được lưu vào file JSON riêng trong
  `artifacts/caption_errors/`. Terminal in đường dẫn file để mở kiểm tra. Mỗi lần lỗi
  tạo file mới; caption lỗi không được đưa vào bảng captions hoặc BM25.
- BM25 dùng SQLite FTS5, **tách theo âm tiết Unicode**, chuẩn hóa NFC, giữ dấu
  tiếng Việt; chưa dùng bộ tách từ tiếng Việt. Query nên bằng tiếng Việt có dấu.
- Query được chuyển thành các token literal nối bằng OR để thu hồi rộng;
  không diễn giải toán tử FTS do người dùng nhập. Chưa bỏ stopword, dịch query,
  hiểu từ đồng nghĩa hay rerank. Các caption trùng từ chung có thể lên kết quả.
- `bm25_score` của SQLite **càng nhỏ càng tốt** (thường âm). Chỉ trả ảnh có từ
  khớp, không bổ sung ảnh điểm 0 chỉ để đủ top-k.
- Đường dẫn kết quả tương đối với data-root. Nên đánh giá Recall@K bằng query
  có ảnh đúng trước khi mở rộng. Nhánh dense bên dưới chạy độc lập; notebook
  `hybrid_retrieval.ipynb` hỗ trợ gộp hai nhánh bằng RRF.

### Trạng thái kiểm tra trên máy phát triển

28 kiểm thử đã đạt trong môi trường `legalqa`: gồm 5 kiểm thử manifest,
11 kiểm thử nhánh lexical/generation và 12 kiểm thử dense.
Kiểm thử dùng caption giả lập trong thư mục tạm để xác minh resume, cache hash,
kiểm tra cấu hình, toàn vẹn dữ liệu và BM25 thật của SQLite; không tạo caption
giả vào dữ liệu dự án. Database thực tế đã có 5.000 caption tiếng Anh và 5.000
bản ghi trong BM25, khớp ID/đường dẫn/hash của manifest. Chưa đánh giá Recall@K.

Máy có RTX 4060 Laptop 8 GB. Môi trường `legalqa` đã được người dùng xác nhận
có PyTorch `2.12.1+cu130`, CUDA khả dụng. Hãy kích hoạt đúng môi trường này
trước khi cài dependencies hoặc chạy caption. Python hệ thống mà agent dùng
để kiểm thử là một môi trường khác, có PyTorch bản CPU.

Kiểm tra interpreter và CUDA trong terminal đang chạy:

```powershell
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.cuda.is_available())"
```

`--device cpu` vẫn được hỗ trợ nhưng chậm và dùng nhiều RAM hơn.

## Dense embedding và retrieval bằng SigLIP 2 Large

Checkpoint: [google/siglip2-large-patch16-384](https://huggingface.co/google/siglip2-large-patch16-384),
FixRes 384 × 384, patch 16, vector ảnh/văn bản **1.024 chiều**. Tên Large ~303M
chỉ quy mô vision encoder; checkpoint gồm cả text encoder khoảng 0,9B tham số,
file trọng số khoảng 3,53 GB. Nhánh này encode **ảnh gốc**, không encode caption.

Chạy từ thư mục gốc dự án, dùng môi trường `legalqa` có PyTorch CUDA:

```powershell
conda activate legalqa
python -m pip install -r requirements-dense.txt
# Thử 24 ảnh; --limit tính số bản ghi MỚI trong lần chạy:
python -m src.dense build --device cuda --batch-size 8 --limit 24
python -m src.dense search "a person wearing a red shirt near the refrigerator" --device cuda --allow-partial --top-k 10
# Tiếp tục toàn bộ các ảnh còn thiếu:
python -m src.dense build --device cuda --batch-size 8
python -m src.dense search "a person washing dishes at the sink" --device cuda --top-k 10
```

Không cần tạo chỉ mục ANN riêng: với 5.000 ảnh, NumPy tính cosine chính xác trên
ma trận 5.000 × 1.024, khoảng 20,5 MB vector float32. Kết quả JSON gồm `imageID`,
`file_path` tương đối với data-root và `cosine_score` **càng lớn càng tốt**.
Đây là độ tương đồng, không phải xác suất. Dùng câu mô tả ngắn; có thể thử tiếng
Anh và tiếng Việt rồi đánh giá bằng tập query có đáp án. Chưa có dịch query,
rerank, fusion với BM25 hoặc cam kết chất lượng truy hồi.

### Lưu, resume và kiểm tra tính nhất quán

- `artifacts/dense.sqlite` chứa `embeddings` và `dense_metadata`; lưu vector
  float32 little-endian được chuẩn hóa L2, cùng ID/đường dẫn/SHA-256. Database
  lexical hiện có không được dùng để lưu dense vectors.
- Commit mỗi batch. Nếu lỗi hoặc dừng giữa chừng, các batch đã commit vẫn còn;
  chạy lại cùng lệnh để tiếp tục. Nếu thiếu VRAM, giảm `--batch-size` xuống 4
  hoặc 1. Không chạy nhiều tiến trình ghi vào cùng dense database.
- Kiểm tra SHA-256 của mọi ảnh, kể cả ảnh đã có vector, trước khi inference.
  Khi manifest thay đổi, ảnh thiếu hoặc khác đường dẫn/hash được encode lại;
  các bản ghi ngoài manifest không được đưa vào kết quả tìm kiếm. Không sửa
  dữ liệu ảnh trong lúc pipeline đang chạy.
- Ảnh có byte giống hệt được tái sử dụng vector theo SHA-256, nhưng từng
  `imageID` vẫn có bản ghi riêng. Không tái sử dụng cho ảnh chỉ gần giống nhau.
- Mặc định ghim commit `1b426889ea62b5a72bf9839009a1b184bfc9c178`; processor và
  cả hai encoder dùng cùng commit. `--revision` chỉ nhận full commit hash.
  Đổi commit/cấu hình/phiên bản Transformers hoặc dtype khi build phải dùng
  database khác qua `--db`, tránh trộn vector từ các cấu hình khác nhau.
- CUDA dùng BF16 nếu hỗ trợ, nếu không dùng FP16; CPU dùng FP32. Có thể search
  trên CPU với database tạo bằng CUDA; sai khác số học nhỏ có thể đổi thứ hạng
  của các ảnh có điểm rất sát nhau. Build resume cần cùng dtype đã lưu.
- Văn bản được chuyển chữ thường, padding đến 64 token và truncate tại 64
  token theo cấu hình SigLIP 2. Không thêm prompt phân loại ảnh vào query.
- Search mặc định yêu cầu mọi dòng manifest có vector khớp đường dẫn/hash.
  `--allow-partial` chỉ tìm trong phần đã hoàn thành. Search không đọc lại byte
  ảnh; nếu đã chỉnh ảnh, cần tạo lại manifest và chạy build trước khi search.
- `--manifest`, `--data-root` (build), `--db`, `--cache-dir`, `--device` hỗ trợ
  đổi vị trí/chế độ chạy. Đường dẫn mặc định neo tại thư mục dự án; đường dẫn
  tùy chỉnh tính theo thư mục chạy lệnh. Trọng số tải vào `artifacts/hf_cache/`.
  Sau khi tải xong, thêm `--local-files-only` để chạy hoàn toàn từ cache, không
  cần truy cập Hugging Face.
- Dùng `python -m src.dense ...` để chạy. Transformers 4 trả feature Tensor,
  Transformers 5 trả `pooler_output`; pipeline xử lý cả hai API.

Kiểm thử dùng vector giả lập trong thư mục tạm để kiểm tra resume, cache hash,
rollback batch lỗi, vector không hợp lệ/hỏng, thay đổi cấu hình/ảnh, manifest
chưa đủ và cosine ranking thực bằng NumPy. Chúng không thay thế đánh giá
chất lượng mô hình trên query có đáp án.

### Kết quả chạy thực ngày 21/11/2026 (Asia/Bangkok)

- Hoàn thành **5.000/5.000 ảnh**, gồm 4.968 hash ảnh khác nhau, trên RTX 4060
  Laptop 8 GB, BF16, batch size 8, Transformers 5.15.0 trong `legalqa`.
- `artifacts/dense.sqlite`: 23.646.208 byte; ma trận vector 5.000 × 1.024.
  Mọi ID/đường dẫn/hash khớp manifest, vector đều hữu hạn; norm L2 lưu thực tế
  từ 0,99999988 đến 1,00000012. SQLite `integrity_check` trả `ok`.
- Đã thử truy vấn tiếng Anh và tiếng Việt trên toàn bộ tập. Chạy lại build
  xác nhận 0 vector mới, không nạp mô hình. BM25 vẫn có đủ 5.000 bản ghi.
- Báo cáo cấu hình, thống kê vector và top-5 của hai query thử nằm trong
  `artifacts/dense_validation.json`. Chưa có nhãn query để tính Recall@K.

## Notebook kiểm chứng lexical + dense

Mở `hybrid_retrieval.ipynb` và chọn kernel/interpreter **legalqa**. Chạy từ trên
xuống một lần, sau đó chỉ sửa `QUERY` trong ô cuối và chạy lại ô đó. Mô hình và
ma trận vector được giữ trong RAM; không cần nạp lại SigLIP cho mỗi query.
Notebook hiển thị top 10 ảnh trong lưới 2 cột, kèm ID, đường dẫn, caption,
điểm BM25/cosine/RRF và thứ hạng ở từng nhánh.

- Mặc định lấy 200 ứng viên mỗi nhánh, gộp bằng weighted Reciprocal Rank Fusion
  với `RRF_K = 60`, hai trọng số bằng 1; trả top 10. Không cộng trực tiếp hai
  loại điểm BM25 và cosine. Các tham số chưa được tối ưu bằng query có nhãn.
- `LEXICAL_QUERY = None` dùng chung `QUERY`. Caption hiện tại bằng tiếng Anh;
  có thể đặt từ khóa riêng như `LEXICAL_QUERY = "red shirt"` để thử lexical.
- `VIEW = "hybrid"`, `"lexical"` hoặc `"dense"` chọn nhánh hiển thị;
  `SHOW_COMPARISON = True` hiện thêm kết quả của hai nhánh riêng.
- Notebook đọc database ở chế độ read-only, kiểm tra index khớp manifest,
  dùng checkpoint từ cache và không sửa ảnh gốc. Output mẫu được lưu sẵn.
- Nếu môi trường chưa có kernel: `python -m pip install ipykernel` sau khi
  kích hoạt `legalqa`; các dependencies còn lại ở `requirements-dense.txt`.
