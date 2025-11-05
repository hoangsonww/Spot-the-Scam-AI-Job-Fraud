import pandas as pd

from spot_scam.data.ingest import clean_text, concatenate_text_fields


def test_clean_text_removes_html_and_urls():
    raw = "<p>Hello <a href='https://example.com'>World</a>!</p>"
    cleaned = clean_text(raw)
    assert "<" not in cleaned
    assert "http" not in cleaned


def test_concatenate_text_fields_handles_missing():
    df = pd.DataFrame(
        {
            "title": ["Engineer"],
            "description": [None],
            "requirements": ["Python"],
        }
    )
    text = concatenate_text_fields(df.loc[0], ["title", "description", "requirements"])
    assert "Engineer" in text and "Python" in text
