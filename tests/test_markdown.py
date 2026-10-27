from pathlib import Path

from citewise.ingest.chunker import chunk_document
from citewise.ingest.markdown import parse_markdown, slugify, strip_markup

FIX = Path(__file__).parent / "fixtures" / "handbook"


def test_mdx_components_and_imports_are_removed_but_text_is_kept():
    doc = parse_markdown((FIX / "people/expenses.mdx").read_text(), "fallback")
    text = doc.text
    assert doc.title == "Spending money"
    assert "import" not in text
    assert "CompensationCalculator" not in text and "<" not in text
    assert "Spend company money the way you would spend your own." in text  # inner text of a component
    assert "card limit of $4,000 per month" in text  # bold markers stripped
    assert "finance page" in text and "/handbook/finance" not in text  # link text kept, URL dropped


def test_liquid_tags_comments_and_permalink():
    doc = parse_markdown((FIX / "travel.md").read_text(), "fallback")
    assert "Internal note for editors" not in doc.text
    assert "{%" not in doc.text
    assert "#travel-help" in doc.text
    assert "reimbursement page" in doc.text
    assert doc.permalink == "/travel-and-expenses/"


def test_headings_build_a_breadcrumb_and_h1_becomes_title():
    doc = parse_markdown((FIX / "engineering/incidents.md").read_text(), "fallback")
    assert doc.title == "Handling an incident"
    assert [s.headings for s in doc.sections] == [
        [],
        ["Severity"],
        ["Useful commands"],
        ["After the incident"],
    ]


def test_hash_inside_code_fence_is_not_a_heading():
    doc = parse_markdown((FIX / "engineering/incidents.md").read_text(), "fallback")
    code = next(b for s in doc.sections for b in s.blocks if b.startswith("```"))
    assert "# this comment is not a heading" in code
    assert all("this comment" not in h for s in doc.sections for h in s.headings)


def test_redirect_pages_are_skipped():
    assert parse_markdown((FIX / "old-page.md").read_text(), "x").skip_reason == "redirect"


def test_nested_headings():
    raw = "---\ntitle: T\n---\n\nIntro text that is long enough to count as content for this test.\n\n## A\n\nalpha text\n\n### B\n\nbeta text\n\n## C\n\ngamma text"
    doc = parse_markdown(raw, "x")
    assert [s.headings for s in doc.sections] == [[], ["A"], ["A", "B"], ["C"]]


def test_strip_markup_handles_multiline_jsx():
    raw = 'Before <Widget\n  title="Hello"\n  size={3}\n/> after'
    assert strip_markup(raw) == "Before Hello after"


def test_slugify_matches_github_anchors():
    assert slugify("Court leave (jury duty)") == "court-leave-jury-duty"
    assert slugify("What's new?") == "whats-new"


def test_chunks_respect_size_and_keep_lists_whole():
    doc = parse_markdown((FIX / "people/time-off.md").read_text(), "x")
    pieces = chunk_document(doc, max_words=60, min_words=10)
    assert all(p.words <= 60 for p in pieces)
    leave = next(p for p in pieces if "Under 6 months" in p.text)
    assert "Over 12 months" in leave.text  # the list was not split across chunks


def test_long_paragraph_is_split_on_sentences():
    long_para = " ".join(f"Sentence number {i} has a few words in it." for i in range(60))
    doc = parse_markdown(f"---\ntitle: Long\n---\n\n## Part\n\n{long_para}", "x")
    pieces = chunk_document(doc, max_words=80, min_words=10)
    assert len(pieces) > 3
    assert all(p.words <= 80 for p in pieces)
    assert all(p.text.rstrip().endswith(".") for p in pieces)


def test_tiny_sections_are_merged_with_siblings():
    raw = "---\ntitle: T\n---\n\n## Policy\n\n### One\n\nshort one\n\n### Two\n\nshort two\n"
    pieces = chunk_document(
        parse_markdown(raw + "\n" + "filler words " * 10, "x"), max_words=100, min_words=20
    )
    assert len(pieces) == 1
    assert pieces[0].headings == ["Policy"]
    assert "One\nshort one" in pieces[0].text and "Two\nshort two" in pieces[0].text


def test_hard_wrapped_lines_are_joined_but_lists_and_tables_keep_their_lines():
    raw = (
        "---\ntitle: T\n---\n\nSubmit your timecard by the end of\nthe day on Friday.\n\n"
        "- First item that wraps\n  onto a second line\n- Second item\n\n| a | b |\n| - | - |\n| 1 | 2 |\n"
    )
    blocks = parse_markdown(raw, "x").sections[0].blocks
    assert blocks[0] == "Submit your timecard by the end of the day on Friday."
    assert blocks[1] == "- First item that wraps onto a second line\n- Second item"
    assert blocks[2] == "| a | b |\n| - | - |\n| 1 | 2 |"
