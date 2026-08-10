"""ReportEngine：报告生成流水线。

模板选择 → 章节装配 → HTML / Markdown 渲染 → PDF 导出。
支持将 ForumEngine 聚合结果按模板自动成稿。
"""
from __future__ import annotations

import asyncio
import html
import os
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..config import Settings

_DEFAULT_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<style>
  body { font-family: "Microsoft YaHei", "PingFang SC", sans-serif; max-width: 860px;
         margin: 24px auto; padding: 0 16px; color: #222; line-height: 1.7; }
  h1 { border-bottom: 3px solid #2f6fed; padding-bottom: 8px; }
  h2 { color: #2f6fed; margin-top: 28px; }
  .meta { color: #888; font-size: 13px; }
  .agent-card { border: 1px solid #e3e8f0; border-radius: 8px; padding: 12px 16px; margin: 10px 0; }
  .agent-card h3 { margin: 4px 0; color: #333; }
  .tag { display: inline-block; background: #eef3ff; color: #2f6fed; border-radius: 4px;
         padding: 1px 8px; font-size: 12px; margin-right: 6px; }
  .risk { color: #c0392b; }
  .ok { color: #27ae60; }
  .conclusion { background: #f7f9fc; border-left: 4px solid #2f6fed; padding: 10px 14px; }
</style>
</head>
<body>
<h1>{{ title }}</h1>
<p class="meta">任务 ID：{{ task_id }} ｜ 查询：{{ query }} ｜ 耗时：{{ elapsed }}s ｜ 成功率：{{ success_rate }}%</p>
{% for section in sections %}
<h2>{{ section.title }}</h2>
{{ section.body | safe }}
{% endfor %}
{% if conclusion %}
<div class="conclusion"><strong>总结：</strong>{{ conclusion }}</div>
{% endif %}
</body>
</html>
"""


class ReportEngine:
    """报告流水线：聚合结果 → 章节装配 → HTML/Markdown → PDF。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._env = Environment(
            loader=FileSystemLoader("templates"),
            autoescape=select_autoescape(["html"]),
        )
        self._env_ok = os.path.isdir("templates")

    def build_sections(self, summary: dict[str, Any]) -> list[dict[str, str]]:
        """把 ForumEngine 聚合结果转成章节（HTML 片段）。"""
        sections: list[dict[str, str]] = []
        agent_results: dict[str, Any] = summary.get("agent_results", {})
        for name, result in agent_results.items():
            body_parts = [f'<span class="tag">{name}</span>']
            for key, value in result.items():
                if key in ("agent",):
                    continue
                if isinstance(value, str):
                    body_parts.append(f"<p><strong>{key}：</strong>{html.escape(value)}</p>")
                elif isinstance(value, dict):
                    body_parts.append(f"<p><strong>{key}：</strong>{html.escape(str(value))}</p>")
                else:
                    body_parts.append(f"<p><strong>{key}：</strong>{html.escape(str(value))}</p>")
            sections.append({"title": f"{name} Agent 分析", "body": "".join(body_parts)})
        if not sections:
            sections.append({"title": "分析结果", "body": "<p>（无可用结果）</p>"})
        return sections

    def _render_html(self, title: str, summary: dict[str, Any], sections: list[dict[str, str]]) -> str:
        if self._env_ok:
            try:
                tmpl = self._env.get_template("report.html")
                return tmpl.render(
                    title=title,
                    task_id=summary.get("task_id", ""),
                    query=summary.get("query", ""),
                    elapsed=summary.get("elapsed_seconds", 0),
                    success_rate=summary.get("success_rate", 0),
                    sections=sections,
                    conclusion=summary.get("conclusion", ""),
                )
            except Exception:
                pass
        # 内置兜底模板
        env = Environment(autoescape=select_autoescape(["html"]))
        tmpl = env.from_string(_DEFAULT_TEMPLATE)
        return tmpl.render(
            title=title,
            task_id=summary.get("task_id", ""),
            query=summary.get("query", ""),
            elapsed=summary.get("elapsed_seconds", 0),
            success_rate=summary.get("success_rate", 0),
            sections=sections,
            conclusion=summary.get("conclusion", ""),
        )

    def render_markdown(self, title: str, summary: dict[str, Any], sections: list[dict[str, str]]) -> str:
        lines = [f"# {title}", ""]
        lines.append(
            f"> 任务：{summary.get('task_id', '')} ｜ 查询：{summary.get('query', '')} "
            f"｜ 耗时：{summary.get('elapsed_seconds', 0)}s ｜ 成功率：{summary.get('success_rate', 0)}%"
        )
        for sec in sections:
            # 去 HTML 标签做纯文本 Markdown
            body = sec["body"]
            import re

            body = re.sub(r"<[^>]+>", " ", body)
            body = re.sub(r"\s+", " ", body).strip()
            lines.append(f"## {sec['title']}")
            lines.append("")
            lines.append(body)
            lines.append("")
        if summary.get("conclusion"):
            lines.append(f"## 总结\n\n{summary['conclusion']}")
        return "\n".join(lines)

    def render_pdf(self, html_content: str, out_path: str) -> str:
        """HTML → PDF（reportlab 简化渲染，支持中文）。"""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        style = ParagraphStyle(
            "cn",
            fontName="STSong-Light",
            fontSize=10.5,
            leading=16,
            wordWrap="CJK",
        )
        doc = SimpleDocTemplate(out_path, pagesize=A4, topMargin=20 * mm, bottomMargin=20 * mm)
        # 抽取正文文本
        import re

        text = re.sub(r"<style>.*?</style>", "", html_content, flags=re.S)
        text = re.sub(r"<[^>]+>", "\n", text)
        text = re.sub(r"\n{2,}", "\n", text).strip()
        story: list[Any] = []
        for para in text.split("\n"):
            para = para.strip()
            if not para:
                continue
            story.append(Paragraph(html.escape(para), style))
            story.append(Spacer(1, 4))
        doc.build(story)
        return out_path

    async def generate(
        self, summary: dict[str, Any], out_dir: str, title: str | None = None
    ) -> dict[str, str]:
        """生成 HTML / Markdown / PDF 三格式报告，返回路径映射。"""
        # 模板渲染与 PDF 制作为 CPU 任务，放入线程池避免阻塞事件循环
        return await asyncio.to_thread(self._generate_sync, summary, out_dir, title)

    def _generate_sync(
        self, summary: dict[str, Any], out_dir: str, title: str | None
    ) -> dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        title = title or f"舆情分析报告-{summary.get('task_id', 'report')}"
        sections = self.build_sections(summary)
        html_content = self._render_html(title, summary, sections)

        base = os.path.join(out_dir, summary.get("task_id", "report"))
        html_path = f"{base}.html"
        md_path = f"{base}.md"
        pdf_path = f"{base}.pdf"

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.render_markdown(title, summary, sections))
        self.render_pdf(html_content, pdf_path)

        return {"html": html_path, "markdown": md_path, "pdf": pdf_path}
