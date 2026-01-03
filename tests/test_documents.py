from langchain_core.documents import Document

from paperfind.documents import extract_title_and_abstract


def test_extract_title_and_abstract_prefers_metadata() -> None:
    doc = Document(
        page_content="Ignored title\n\nIgnored abstract",
        metadata={"title": "Meta Title", "abstract": "Meta Abstract"},
    )

    title, abstract = extract_title_and_abstract(doc)

    assert title == "Meta Title"
    assert abstract == "Meta Abstract"


def test_extract_title_and_abstract_from_content() -> None:
    doc = Document(
        page_content="Content Title\n\nContent abstract.",
        metadata={},
    )

    title, abstract = extract_title_and_abstract(doc)

    assert title == "Content Title"
    assert abstract == "Content abstract."


def test_extract_title_and_abstract_strips_prefix_and_tags() -> None:
    content = "Title\n\nAbstract: Some abstract text.\n\nTags: tag1, tag2"
    doc = Document(page_content=content, metadata={})

    title, abstract = extract_title_and_abstract(doc)

    assert title == "Title"
    assert abstract == "Some abstract text."


def test_extract_title_and_abstract_handles_single_newline() -> None:
    content = "Title\nAbstract line 1\nline 2"
    doc = Document(page_content=content, metadata={})

    title, abstract = extract_title_and_abstract(doc)

    assert title == "Title"
    assert abstract == "Abstract line 1\nline 2"


def test_extract_title_and_abstract_handles_empty_abstract() -> None:
    doc = Document(page_content="Only Title", metadata={})

    title, abstract = extract_title_and_abstract(doc)

    assert title == "Only Title"
    assert abstract is None


def test_extract_title_and_abstract_handles_empty_content() -> None:
    doc = Document(page_content="", metadata={})

    title, abstract = extract_title_and_abstract(doc)

    assert title == ""
    assert abstract is None


def test_extract_title_and_abstract_handles_whitespace_only() -> None:
    doc = Document(page_content="   \n\n   ", metadata={})

    title, abstract = extract_title_and_abstract(doc)

    assert title == ""
    assert abstract is None


def test_extract_title_and_abstract_metadata_overrides_empty_content() -> None:
    doc = Document(
        page_content="",
        metadata={"title": "Meta Title", "abstract": "Meta Abstract"},
    )

    title, abstract = extract_title_and_abstract(doc)

    assert title == "Meta Title"
    assert abstract == "Meta Abstract"


def test_extract_title_and_abstract_partial_metadata() -> None:
    """Metadata title with content-based abstract."""
    doc = Document(
        page_content="Ignored\n\nContent abstract here.",
        metadata={"title": "Meta Title"},
    )

    title, abstract = extract_title_and_abstract(doc)

    assert title == "Meta Title"
    assert abstract == "Content abstract here."
