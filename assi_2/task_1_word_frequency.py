from collections import Counter
from pathlib import Path
import re


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "assignment_2_corpus.txt"
TARGET_WORDS = ("and", "to", "arjun", "arjuna")


def normalize_text(text: str) -> list[str]:
    """Lowercase and extract word-like tokens from the raw text."""
    cleaned = re.sub(r"<[^>]+>", " ", text)
    return re.findall(r"[\w']+", cleaned.lower())


def count_target_words(tokens: list[str]) -> dict[str, int]:
    counts = Counter(tokens)
    arjun_count = counts["arjun"] + counts["arjuna"]
    return {
        "and": counts["and"],
        "to": counts["to"],
        "arjun": arjun_count,
    }


def main() -> None:
    text = INPUT_FILE.read_text(encoding="utf-8")
    tokens = normalize_text(text)
    counts = count_target_words(tokens)

    print(f"Total tokens: {len(tokens)}")
    print(f"Count(and): {counts['and']}")
    print(f"Count(to): {counts['to']}")
    print(f"Count(arjun/arjuna): {counts['arjun']}")


if __name__ == "__main__":
    main()
