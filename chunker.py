"""把模型吐出的字流切成可以拿去合成的小段。"""

from __future__ import annotations

# 句末 + 分号/逗号；不做「满 N 字强制切」
SENTENCE_ENDS = "。！？\n.?!；，,"


class Chunker:
    def __init__(self) -> None:
        self._buffer = ""

    def push(self, delta: str) -> list[str]:
        self._buffer += delta
        return self._pull(final=False)

    def finish(self) -> list[str]:
        return self._pull(final=True)

    def _pull(self, final: bool) -> list[str]:
        pieces: list[str] = []
        while True:
            piece = self._next(final)
            if piece is None:
                return pieces
            if piece.strip():
                pieces.append(piece)

    def _next(self, final: bool) -> str | None:
        if not self._buffer:
            return None
        ends = [self._buffer.find(mark) for mark in SENTENCE_ENDS]
        ends = [index for index in ends if index >= 0]
        if ends:
            cut = min(ends) + 1
        elif final:
            cut = len(self._buffer)
        else:
            return None
        piece, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return piece
