"""Notebook controls for manual QA and explicit DRES submissions."""

import io
import json

from src.dres_client import to_item_name
from src.qa import evidence_image, load_draft, make_draft, save_draft


def _display_evidence(data_root, row, crop=None):
    from IPython.display import Image, display

    image = evidence_image(data_root, row, crop)
    try:
        print(f"{row['imageID']} | {image.width} × {image.height} px")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        display(Image(data=buffer.getvalue(), width=1000 if crop is not None else min(image.width, 1400)))
    finally:
        image.close()


def manual_qa_panel(retrieval, rows, data_root, draft_path, question="", task_label=""):
    import ipywidgets as widgets
    from IPython.display import display

    rows_by_id = {row["imageID"]: row for row in rows}
    candidates = list(dict.fromkeys(hit["imageID"] for hit in retrieval["hybrid"]))
    question_input = widgets.Textarea(value=question, description="Câu hỏi:",
                                     layout=widgets.Layout(width="95%", height="80px"))
    label_input = widgets.Text(value=task_label, description="Tên task:", placeholder="QA01 (nếu đã biết)")
    evidence = widgets.Dropdown(options=candidates, description="Ảnh:",
                                layout=widgets.Layout(width="600px"))
    crop_input = widgets.Text(description="Crop:", placeholder="Để trống hoặc left,top,right,bottom",
                              layout=widgets.Layout(width="600px"))
    answer_input = widgets.Text(value="", description="Đáp án:", placeholder="Tự đọc ảnh và nhập đáp án ngắn",
                                layout=widgets.Layout(width="95%"))
    preview_button = widgets.Button(description="Xem ảnh / crop")
    save_button = widgets.Button(description="Lưu bản nháp QA", button_style="info")
    preview_output = widgets.Output()
    status = widgets.Output()

    def preview(_=None):
        with preview_output:
            preview_output.clear_output(wait=True)
            try:
                if not evidence.value:
                    raise ValueError("Chưa có ảnh hybrid để chọn. Hãy chạy retrieval trước.")
                crop = tuple(int(value.strip()) for value in crop_input.value.split(",")) if crop_input.value.strip() else None
                _display_evidence(data_root, rows_by_id[evidence.value], crop)
            except (OSError, ValueError) as exc:
                print(str(exc))

    def save(_):
        with status:
            status.clear_output(wait=True)
            try:
                draft = make_draft(question_input.value, retrieval["query"], evidence.value or "",
                                   answer_input.value, rows, label_input.value)
                save_draft(draft_path, draft, rows)
                print(f"Đã lưu: {draft_path}")
                print(f"Đáp án: {draft['answer']} | Bằng chứng: {draft['imageID']}")
                print("Mở dres_submission.ipynb và bấm Nạp bản nháp QA để nộp.")
            except (OSError, ValueError) as exc:
                print(str(exc))

    def change_evidence(change):
        crop_input.value = ""
        answer_input.value = ""
        preview()

    evidence.observe(change_evidence, names="value")
    preview_button.on_click(preview)
    save_button.on_click(save)
    panel = widgets.VBox([
        widgets.Label(value=f"Query tìm ảnh: {retrieval['query']}"),
        question_input, label_input, evidence, crop_input, preview_button,
        preview_output, answer_input, save_button, status,
    ])
    display(panel)
    preview()
    return panel


def dres_panel(client, rows, data_root, draft_path):
    import ipywidgets as widgets
    from IPython.display import display

    rows_by_id = {row["imageID"]: row for row in rows}
    evaluation = widgets.Dropdown(options=[("Chọn buổi thi", "")], description="Buổi thi:",
                                  layout=widgets.Layout(width="95%"))
    refresh_evaluations = widgets.Button(description="Tải buổi thi")
    refresh_task = widgets.Button(description="Xem task hiện tại")
    label_input = widgets.Text(description="Tên task:", placeholder="Nhập đúng tên trên Viewer: QA01 / KIS01",
                               layout=widgets.Layout(width="95%"))
    kind = widgets.Dropdown(options=["QA", "KIS"], value="QA", description="Loại bài:")
    answer_input = widgets.Textarea(description="Đáp án QA:", layout=widgets.Layout(width="95%", height="65px"))
    image_input = widgets.Text(description="Ảnh KIS:", placeholder="day1_Kitchen_08_102 hoặc 08_102.jpg",
                               layout=widgets.Layout(width="95%"))
    load_button = widgets.Button(description="Nạp bản nháp QA")
    preview_image = widgets.Button(description="Xem ảnh KIS")
    prepare_button = widgets.Button(description="Xem trước bài nộp", button_style="info")
    submit_button = widgets.Button(description="Nộp lên DRES", button_style="danger", disabled=True)
    history_button = widgets.Button(description="Xem lịch sử cục bộ")
    task_output, draft_output, preview_output, result_output = (widgets.Output() for _ in range(4))
    state = {"prepared": None}

    def invalidate(_=None):
        state["prepared"] = None
        submit_button.disabled = True
        preview_output.clear_output()

    def show_kind(_=None):
        answer_input.disabled = kind.value != "QA"
        image_input.disabled = kind.value != "KIS"
        preview_image.disabled = kind.value != "KIS"
        invalidate()

    for field in (evaluation, label_input, answer_input, image_input):
        field.observe(invalidate, names="value")
    kind.observe(show_kind, names="value")

    def refresh(_):
        invalidate()
        with task_output:
            task_output.clear_output(wait=True)
            try:
                active = [item for item in client.list_evaluations() if item.get("status") == "ACTIVE"]
                previous = evaluation.value
                evaluation.options = [("Chọn buổi thi", "")] + [
                    (f"{item.get('name', '')} | {item['id']}", item["id"]) for item in active]
                evaluation.value = previous if any(item["id"] == previous for item in active) else ""
                print(f"Có {len(active)} buổi thi ACTIVE. Chọn đúng buổi thi rồi xem task.")
            except Exception as exc:
                print(str(exc))

    def show_task(_):
        invalidate()
        with task_output:
            task_output.clear_output(wait=True)
            try:
                if not evaluation.value:
                    raise ValueError("Hãy chọn buổi thi.")
                task = client.current_task(evaluation.value)
                print(json.dumps(task, ensure_ascii=False, indent=2) if task else "Chưa có task (hoặc evaluation không còn tồn tại).")
                print("Đọc câu hỏi trên DRES Viewer; nhập đúng tên task và chọn loại bài bên dưới.")
            except Exception as exc:
                print(str(exc))

    def load(_):
        invalidate()
        with draft_output:
            draft_output.clear_output(wait=True)
            try:
                draft = load_draft(draft_path, rows)
                kind.value = "QA"
                answer_input.value = draft["answer"]
                label_input.value = draft.get("task_label", "")
                print(f"Câu hỏi bản nháp: {draft['question']}")
                print(f"Query: {draft['query']} | Tạo lúc: {draft.get('created_at_utc', '')}")
                print("Đối chiếu câu hỏi và tên task với Viewer trước khi nộp.")
                _display_evidence(data_root, rows_by_id[draft["imageID"]])
            except (OSError, ValueError) as exc:
                print(str(exc))

    def show_image(_):
        with draft_output:
            draft_output.clear_output(wait=True)
            try:
                image_id = to_item_name(image_input.value)
                if image_id not in rows_by_id:
                    raise ValueError("Ảnh KIS không có trong manifest.")
                _display_evidence(data_root, rows_by_id[image_id])
            except (OSError, ValueError) as exc:
                print(str(exc))

    def prepare(_):
        invalidate()
        with preview_output:
            try:
                value = answer_input.value if kind.value == "QA" else image_input.value
                if kind.value == "KIS" and to_item_name(value) not in rows_by_id:
                    raise ValueError("Ảnh KIS không có trong manifest.")
                prepared = client.prepare(evaluation.value, label_input.value, kind.value, value)
                print(f"Evaluation: {prepared.evaluation_id} | Task: {prepared.task_label} | {prepared.kind}")
                print("Task API:", prepared.marker)
                print(json.dumps(prepared.payload, ensure_ascii=False, indent=2))
                marker = json.loads(prepared.marker)
                if not any(key in marker for key in ("id", "taskId", "name", "taskName")):
                    print("API chỉ trả loại task: cần tự đối chiếu tên task trên Viewer ngay trước khi bấm nộp.")
                state["prepared"] = prepared
                submit_button.disabled = False
            except Exception as exc:
                print(str(exc))

    def submit(_):
        prepared = state["prepared"]
        invalidate()
        with result_output:
            result_output.clear_output(wait=True)
            try:
                if prepared is None:
                    raise ValueError("Hãy xem trước bài nộp trước.")
                result = client.submit(prepared)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                if result["submission"] == "INDETERMINATE" or result["http_status"] == 202:
                    print("DRES đã nhận bài, đang chờ trọng tài chấm. Theo dõi Viewer/bảng điểm.")
                elif result["submission"] == "CORRECT":
                    print("Đáp án đúng. Không cần nộp thêm cho task này.")
                elif result["submission"] == "WRONG":
                    print("Đáp án sai. Xem lại bằng chứng trước khi nộp đáp án khác.")
                elif result["http_status"] == 401:
                    print("Session hết hạn. Chạy lại ô đăng nhập, tạo lại panel và xem trước bài.")
                elif result["state"] == "UNKNOWN":
                    print("Chưa rõ kết quả. Kiểm tra DRES trước khi quyết định bước tiếp theo.")
            except Exception as exc:
                print(str(exc))

    def history(_):
        with result_output:
            result_output.clear_output(wait=True)
            try:
                if client.history_path is None or not client.history_path.exists():
                    print("Chưa có lịch sử nộp.")
                    return
                lines = client.history_path.read_text(encoding="utf-8").splitlines()
                print("\n".join(lines[-20:]))
            except OSError as exc:
                print(str(exc))

    refresh_evaluations.on_click(refresh)
    refresh_task.on_click(show_task)
    load_button.on_click(load)
    preview_image.on_click(show_image)
    prepare_button.on_click(prepare)
    submit_button.on_click(submit)
    history_button.on_click(history)
    show_kind()
    panel = widgets.VBox([
        widgets.HBox([refresh_evaluations, refresh_task]), evaluation, task_output,
        label_input, kind, load_button, answer_input, image_input, preview_image,
        draft_output, prepare_button, preview_output, submit_button, history_button, result_output,
    ])
    display(panel)
    return panel
