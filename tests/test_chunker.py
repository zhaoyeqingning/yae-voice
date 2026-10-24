from chunker import Chunker


def test_first_piece_ends_at_chinese_punctuation():
    chunker = Chunker()
    assert chunker.push("你好。世界") == ["你好。"]
    assert chunker.finish() == ["世界"]


def test_question_exclamation_and_newline_also_end_a_piece():
    chunker = Chunker()
    assert chunker.push("在吗？\n好！") == ["在吗？", "好！"]


def test_english_and_comma_semicolon_also_end_a_piece():
    chunker = Chunker()
    assert chunker.push("Hi. OK! What? 嗯，对；行") == [
        "Hi.",
        " OK!",
        " What?",
        " 嗯，",
        "对；",
    ]
    assert chunker.finish() == ["行"]


def test_first_piece_waits_for_punctuation():
    chunker = Chunker()
    text = "甲" * 20 + "乙"
    assert chunker.push(text) == []
    assert chunker.finish() == [text]


def test_later_pieces_wait_for_punctuation():
    chunker = Chunker()
    assert chunker.push("你好。") == ["你好。"]
    assert chunker.push("甲" * 25) == []
    assert chunker.finish() == ["甲" * 25]
