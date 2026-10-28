"""Render assistant Markdown to a small, sanitized HTML allow-list."""

from __future__ import annotations

from html import escape
from html.parser import HTMLParser
from urllib.parse import urlsplit

from markdown_it import MarkdownIt

_MARKDOWN = MarkdownIt("gfm-like", {"html": False, "linkify": False})
_ALLOWED = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li",
    "strong", "em", "del", "code", "pre", "blockquote", "a", "table",
    "thead", "tbody", "tr", "th", "td", "hr", "br",
}


class _Sanitizer(HTMLParser):
    """Keep only structural Markdown tags and safe link/code attributes."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []
        self.drop_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in _ALLOWED:
            self.drop_depth += 1
            return
        if self.drop_depth:
            return
        safe_attrs = ""
        if tag == "a":
            href = dict(attrs).get("href") or ""
            scheme = urlsplit(href).scheme.lower()
            if not scheme or scheme in {"http", "https", "mailto"}:
                safe_attrs = (
                    f' href="{escape(href, quote=True)}" '
                    'rel="nofollow noopener noreferrer"'
                )
        elif tag == "code":
            classes = (dict(attrs).get("class") or "").split()
            language = next(
                (
                    item
                    for item in classes
                    if item.startswith("language-") and item[9:].isalnum()
                ),
                None,
            )
            if language:
                safe_attrs = f' class="{escape(language, quote=True)}"'
        self.parts.append(f"<{tag}{safe_attrs}>")

    def handle_endtag(self, tag: str) -> None:
        if tag not in _ALLOWED:
            self.drop_depth = max(0, self.drop_depth - 1)
        elif not self.drop_depth:
            self.parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if not self.drop_depth:
            self.parts.append(escape(data))

    def handle_entityref(self, name: str) -> None:
        if not self.drop_depth:
            self.parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if not self.drop_depth:
            self.parts.append(f"&#{name};")


def render_safe_markdown(source: str) -> str:
    """Render common Markdown and strip tags/attributes outside the allow-list."""
    parser = _Sanitizer()
    parser.feed(_MARKDOWN.render(source))
    parser.close()
    return "".join(parser.parts)
