"""验证普通文本、代码保护与整页/竖排回填共用非 CJK 墨迹词界规则。"""

from typing import Any

import pytest
from docvortex.document.pdf import Bbox
from mineru.backend.analysis.pdf.text import native
from mineru.backend.analysis.pdf.text.models import _AnalyzeSpan
from mineru.types import ContentType


def _char(text: str, index: int, x: float) -> dict[str, Any]:
    """构造短英文或 CJK 字符，故意使 loose 框覆盖真实留白。"""
    return {
        "char": text,
        "char_idx": index,
        "bbox": Bbox([x, 0, x + 10, 10]),
        "tight_bbox": (x, 1, x + 4, 9),
        "origin": (x, 10),
        "writing_angle": 0.0,
        "rotation": 0.0,
        "font": {"name": "Fixture", "size": 10.0, "flags": 0, "weight": 400},
    }


@pytest.mark.parametrize("text, expected", [("AIML", "AI ML"), ("中文", "中文"), ("中A", "中A"), ("A文", "A文")])
def test_short_span_needs_no_reference_samples(text: str, expected: str) -> None:
    """无需同行参考样本即可补短英文，CJK 边界保持旧输出。"""
    positions = [0, 5, 13, 18] if len(text) == 4 else [0, 8]
    chars = [_char(char, index, positions[index]) for index, char in enumerate(text)]
    span = _AnalyzeSpan(ContentType.TEXT, (0, 0, 40, 10), metadata={"chars": chars})
    native.chars_to_content(span, detect_scripts=False)
    assert span.content == expected


def test_code_span_keeps_old_content() -> None:
    """被代码块认领的 span 不启用新增墨迹词界。"""
    span = _AnalyzeSpan(
        ContentType.TEXT,
        (0, 0, 30, 10),
        metadata={"chars": [_char("A", 0, 0), _char("B", 1, 8)], "_native_tight_spacing": False},
    )
    native.chars_to_content(span, detect_scripts=False)
    assert span.content == "AB"


def test_virtual_page_fill_keeps_actual_code_region() -> None:
    """实际页面代码区域的禁用标记不会被虚拟整页文本框覆盖。"""
    from mineru.backend.analysis.pdf.text.content import _protect_code_span_spacing
    from mineru.types import BlockType
    from unittest.mock import Mock

    chars = [_char("A", 0, 0), _char("B", 1, 8)]
    span = _AnalyzeSpan(ContentType.TEXT, (0, 0, 30, 10))
    _protect_code_span_spacing([span], [{"type": BlockType.CODE, "bbox": (0, 0, 0.5, 0.5)}], (100, 100))
    page = Mock()
    page.get_char_count.return_value = 2
    native.txt_spans_extract(
        page,
        [span],
        None,
        1.0,
        [(0, 0, 100, 100, None, None, None, BlockType.TEXT)],
        [],
        page_chars=chars,
        detect_scripts=False,
    )
    assert span.content == "AB"
    assert span.metadata["_native_tight_spacing"] is False


@pytest.mark.parametrize("text", ["AB", "中 文"])
@pytest.mark.parametrize("spacing", [True, False])
def test_vertical_line_fill_uses_tight_spacing(monkeypatch: pytest.MonkeyPatch, text: str, spacing: bool) -> None:
    """旋转行回填共用英文词界及中文生成空格规则，代码仍保留原文。"""
    import math
    from unittest.mock import Mock
    from PIL import Image
    from mineru.types import BlockType

    chars = [_char(char, i, 0) for i, char in enumerate(text)]
    positions = [0, 8] if text == "AB" else [0, 5, 10]
    for char, y in zip(chars, positions):
        char.update(
            bbox=Bbox([40, y, 50, y + 10]),
            tight_bbox=(41, y, 49, y + (4 if text == "AB" else 8)),
            origin=(40, y),
            rotation=math.pi / 2,
            writing_angle=math.pi / 2,
            is_generated=char["char"] == " ",
        )
    line = {"rotation": math.pi / 2, "bbox": Bbox([40, 0, 50, 30]), "spans": [{"text": text, "chars": chars}]}
    monkeypatch.setattr(native, "get_lines_from_chars", Mock(return_value=[line]))
    page = Mock()
    page.get_char_count.return_value = len(chars)
    target = _AnalyzeSpan(ContentType.TEXT, (40, 0, 50, 30), metadata={"_native_tight_spacing": spacing})
    spans = [target, _AnalyzeSpan(ContentType.TEXT, (60, 60, 80, 68)), _AnalyzeSpan(ContentType.TEXT, (60, 80, 80, 88))]
    with Image.new("RGB", (100, 100), "white") as image:
        native.txt_spans_extract(
            page,
            spans,
            image,
            1.0,
            [(0, 0, 100, 100, None, None, None, BlockType.TEXT)],
            [],
            page_chars=chars,
            detect_scripts=False,
        )
    assert target.content == (("A B" if text == "AB" else "中文") if spacing else text)


@pytest.mark.parametrize("generated", [True, False, None])
@pytest.mark.parametrize("spacing", [True, False])
@pytest.mark.parametrize("scripts", [True, False])
def test_generated_cjk_space_materialization(generated: bool | None, spacing: bool, scripts: bool) -> None:
    """Python 回填只忽略已确认生成的中文空格，旧字距不会重加，代码及未知证据保留。"""
    chars = [_char(char, i, i * 5) for i, char in enumerate("卫 星 InSAR 技 术")]
    for char in chars:
        char["tight_bbox"] = tuple(char["bbox"].bbox)
        if generated is not None:
            char["is_generated"] = generated and char["char"] == " "
    span = _AnalyzeSpan(ContentType.TEXT, (0, 0, 250, 10), metadata={"chars": chars, "_native_tight_spacing": spacing})
    native.chars_to_content(span, detect_scripts=scripts)
    assert span.content == ("卫星 InSAR 技术" if generated is True and spacing else "卫 星 InSAR 技 术")
    assert len(chars) == 13


@pytest.mark.parametrize("spacing", [True, False])
def test_generated_cjk_spaces_owned_snapshot_matches_python(spacing: bool) -> None:
    """真实 PDF 关闭后 Rust span 内容物化仍与 Python 一致，同时保留真实中英空格。"""
    from io import BytesIO
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from docvortex.document.pdf import PDFDocument

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(240, 100))
    canvas.setFont("STSong-Light", 12)
    for i, char in enumerate("卫星技术"):
        canvas.drawString(10 + i * 15, 70, char)
    canvas.drawString(10, 30, "中 文 InSAR 技术")
    canvas.save()
    with PDFDocument(stream.getvalue()) as document:
        owner = document[0].get_text_snapshot()
        geometry = document[0].get_chars_with_geometry()
    if owner is None:
        pytest.skip("Python backend")
    spans = [_AnalyzeSpan(ContentType.TEXT, (0.0, 0.0, 240.0, 100.0), metadata={"chars": [], "_native_tight_spacing": spacing})]
    records = native._owned_span_texts(owner, spans, 12.0, False)
    assert records is not None
    reference = _AnalyzeSpan(
        ContentType.TEXT, spans[0].bbox, metadata={"chars": geometry.chars, "_native_tight_spacing": spacing}
    )
    native.chars_to_content(reference, detect_scripts=False)
    assert records[0][0] == reference.content
    assert "中 文 InSAR 技术" in reference.content
    assert ("卫星技术" if spacing else "卫 星 技 术") in reference.content
