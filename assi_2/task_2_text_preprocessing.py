from pathlib import Path
import re


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "assignment_2_lstm_source.txt"
OUTPUT_FILE = BASE_DIR / "task_2_processed_output.txt"

# Lightweight fallback set so the script works even without nltk installed.
FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
    "were", "will", "with",
}


def normalize_tokens(text: str) -> list[str]:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{\{[^}]+\}\}", " ", text)
    text = re.sub(r"\[\[[^\]]+\]\]", " ", text)
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    return [token.lower() for token in text.split() if token.strip()]


def get_stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except Exception:
        return FALLBACK_STOPWORDS


def stem_word(word: str) -> str:
    try:
        from nltk.stem import PorterStemmer

        return PorterStemmer().stem(word)
    except Exception:
        # Small fallback so the script still performs a visible stemming step.
        for suffix in ("ingly", "edly", "ing", "ed", "ly", "ies", "s"):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                if suffix == "ies":
                    return word[:-3] + "y"
                return word[: -len(suffix)]
        return word


def preprocess_text(text: str) -> list[str]:
    stop_words = get_stopwords()
    processed_tokens = []

    for token in normalize_tokens(text):
        if token in stop_words:
            continue
        processed_tokens.append(stem_word(token))

    return processed_tokens


def main() -> None:
    text = INPUT_FILE.read_text(encoding="utf-8")
    processed_tokens = preprocess_text(text)
    OUTPUT_FILE.write_text(" ".join(processed_tokens), encoding="utf-8")

    print(f"Saved {len(processed_tokens)} processed tokens to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
