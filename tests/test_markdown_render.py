from agent.markdown_render import render_safe_markdown


def test_assistant_markdown_renders_formatting_and_gfm_table():
    rendered = render_safe_markdown(
        "**Finding**\n\n- Review the region\n- Score: `0.42`\n\n"
        "| Metric | Value |\n| --- | ---: |\n| Coverage | 20% |"
    )
    assert "<strong>Finding</strong>" in rendered
    assert "<ul>" in rendered and "<li>Review the region</li>" in rendered
    assert "<code>0.42</code>" in rendered
    assert "<table>" in rendered and "Coverage" in rendered


def test_assistant_markdown_removes_unsafe_html_and_links():
    rendered = render_safe_markdown(
        '<script>alert("secret")</script> [bad](javascript:alert(1)) [ok](https://example.org)'
    )
    assert "<script" not in rendered and "alert(\"secret\")" not in rendered
    assert 'href="javascript:' not in rendered
    assert 'href="https://example.org"' in rendered
