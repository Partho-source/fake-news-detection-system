"""validator.py - checks the news text entered by the user."""

MIN_WORDS = 10
MAX_CHARS = 10000

def validate_input(text):
    """Return (is_valid, message)."""
    if text is None or not text.strip():
        return False, "Please enter a news article."
    if len(text) > MAX_CHARS:
        return False, "Article is too long (limit 10000 characters)."
    if len(text.split()) < MIN_WORDS:
        return False, "Article is too short. Enter at least 10 words."
    return True, "OK"
