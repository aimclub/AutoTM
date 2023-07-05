import pytest

from autotm.preprocessing.text_preprocessing import (remove_html,
                                                     process_punkt,
                                                     get_lemma, lemmatize_text_ru)


@pytest.mark.parametrize(
    "input,expected_output",
    [
        (
                '<p style="font-size:18pt">Text size using points.</p><p style="font-size:18px">'
                'Text size using pixels.</p><p style="font-size:larger">Text size using relative sizes.</p>',
                'Text size using points.Text size using pixels.Text size using relative sizes.'
        ),
        (
                '', ''
        )
    ]
)
def test_html_removal(input, expected_output):
    """Test html is removed from text"""
    assert remove_html(input) == expected_output


@pytest.mark.parametrize(
    "input,expected_output",
    [
        (
                "  lots     of     space  ", "lots of space"
        ),
        (
                "No#, punctuation? id4321 removed", "No punctuation removed"
        )
    ]
)
def test_process_punct(input, expected_output):
    """Test punctuation removal is correct"""
    assert process_punkt(input) == expected_output


@pytest.mark.parametrize(
    "input,expected_output",
    [
        ("id19898", "id19898"),
        ("нормальные", "нормальные"),
        ("dogs", "dog"),
        ("toughest", "tough")
    ]
)
def test_get_lemma(input, expected_output):
    """Test lemmatization works as intended"""
    assert get_lemma(input) == expected_output


@pytest.mark.parametrize(
    "input,expected_output",
    [
        ("Эти тексты для пользователя id198890, потому что он пес", "текст пользователь пес"),
        ("Оболочка работает с CPython 2.6+/3.3+ и PyPy 1.9+", "оболочка работать cpython pypy")
    ]
)
def test_full_russian_processing(input, expected_output):
    """Test preprocessing for russian"""
    assert lemmatize_text_ru(input) == expected_output
