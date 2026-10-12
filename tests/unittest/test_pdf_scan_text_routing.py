"""通过真正的 DocVortex 分类器核验 MinerU auto 对扫描补文本与数字背景页的路由。"""

from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from mineru.backend.analysis.pdf import pipeline
from mineru.model.runtime import hybrid as hybrid_runtime


def _image_layer_pdf(kind: str) -> bytes:
    """独立构造背景正文、隐藏搜索层、透明文字和被图像遮住的文字，避免引用另一个仓库的测试。"""
    stream = BytesIO()
    painter = Canvas(stream, pagesize=(400, 300))
    image = Image.new("RGB", (800, 600), (225, 235, 250))
    if kind != "covered":
        painter.drawImage(ImageReader(image), 0, 0, 400, 300)
    painter.saveState()
    if kind == "transparent":
        painter.setFillAlpha(0)
    text = painter.beginText(20, 260)
    text.setFont("Helvetica", 9)
    text.setTextRenderMode(3 if kind == "hidden" else 0)
    for line in [
        "A real document may have a large decorative background image.",
        "A searchable scan must still use the OCR inference pipeline.",
        "Existing invisible text must not be adopted as native body text.",
    ]:
        text.textLine(line)
    painter.drawText(text)
    painter.restoreState()
    if kind == "covered":
        painter.drawImage(ImageReader(image), 0, 0, 400, 300)
    painter.save()
    image.close()
    return stream.getvalue()


@pytest.mark.parametrize(
    "kind,expected", [("background", "txt"), ("hidden", "ocr"), ("transparent", "ocr"), ("covered", "ocr")]
)
@pytest.mark.parametrize("effort", ["flash", "medium"])
def test_real_auto_classification_controls_mineru_model_routing(
    monkeypatch: pytest.MonkeyPatch, kind: str, expected: str, effort: str
) -> None:
    """只替换推理模型加载，真实 PDF 和 classify 决定解析模式及 Flash 原生分支。"""
    calls = []

    def model() -> SimpleNamespace:
        """记录宿主推理初始化，背景 Flash 页应完全跳过该调用。"""
        calls.append("model")
        return SimpleNamespace(device="cpu")

    monkeypatch.setattr(hybrid_runtime, "HybridLocalModelContextSingleton", lambda: SimpleNamespace(get_model=model))
    monkeypatch.setattr(pipeline, "acquire_document", lambda *_: None)
    monkeypatch.setattr(pipeline, "release_document", lambda *_: None)
    monkeypatch.setattr(pipeline, "trim_process_heap", lambda: None)
    state = pipeline._PDFAnalysis()
    try:
        pipeline._prepare_analysis(state, _image_layer_pdf(kind), effort, "auto", None)
        assert state.parse_mode == expected
        assert state.flash_txt_mode is (effort == "flash" and expected == "txt")
        assert bool(calls) is (not state.flash_txt_mode)
    finally:
        pipeline._close_analysis(state)
