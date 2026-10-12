"""MinerU 只通过 DocVortex 公开能力消费 Office 表格链接修复。"""
from pathlib import Path

import pytest
from mineru.parser import parse
from mineru.render import render_html


@pytest.mark.parametrize("suffix", ["docx", "pptx"])
def test_office_table_links_public_parser(suffix: str) -> None:
    """验证当前共享提交从 MinerU Flash 入口导出三种格式时保留链接。"""
    path = Path(__file__).resolve().parents[1] / "fixtures" / "pr5389" / f"table-links.{suffix}"
    result = parse(str(path), tier="flash")
    for output in (result.markdown(), render_html(result.middle_json), str(result.structured_content())):
        assert "https://example.org/one" in output and "https://example.org/two" in output
        assert "javascript:" not in output
