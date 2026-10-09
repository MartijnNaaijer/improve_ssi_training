"""Loading the ETCBC tab-separated corpus files (book, chapter, verse, text)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / 'data' / 'raw'


@dataclass(frozen=True)
class Verse:
    """One line of the corpus. `vid` is unique: a few verses are split over two
    lines with the same reference, so the second gets the suffix '#2'."""
    vid: str
    book: str
    chapter: str
    verse: str
    surface: tuple[str, ...]
    analysis: tuple[str, ...]

    def words(self):
        return zip(self.surface, self.analysis)


def _read_tsv(path: Path) -> list[tuple[str, str, str, str]]:
    rows = []
    with open(path, encoding='utf8') as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 4:
                raise ValueError(f'{path}:{n}: expected 4 columns, got {len(parts)}')
            rows.append(tuple(parts))
    return rows


def load_corpus(version: str = 's12', raw_dir: Path = RAW_DIR) -> list[Verse]:
    """Load the aligned surface and analysis files of a corpus version."""
    in_rows = _read_tsv(raw_dir / f'{version}-in')
    out_rows = _read_tsv(raw_dir / f'{version}-out')
    if len(in_rows) != len(out_rows):
        raise ValueError(f'{version}: {len(in_rows)} input lines vs {len(out_rows)} output lines')

    seen = Counter()
    verses = []
    for (b, c, v, text), (b2, c2, v2, enc) in zip(in_rows, out_rows):
        if (b, c, v) != (b2, c2, v2):
            raise ValueError(f'{version}: misaligned lines {b} {c}:{v} vs {b2} {c2}:{v2}')
        surface, analysis = tuple(text.split()), tuple(enc.split())
        if len(surface) != len(analysis):
            raise ValueError(f'{version}: word count mismatch in {b} {c}:{v}')
        ref = f'{b} {c}:{v}'
        seen[ref] += 1
        vid = ref if seen[ref] == 1 else f'{ref}#{seen[ref]}'
        verses.append(Verse(vid, b, c, v, surface, analysis))
    return verses
