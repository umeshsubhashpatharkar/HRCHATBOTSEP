def simple_chunk_text(text, chunk_size=50, overlap_pct=0.1):
    """
    Split input text into overlapping word-based chunks.

    Args:
        text (str): Input text to chunk.
        chunk_size (int): Number of words per chunk (default 50).
        overlap_pct (float): Fraction (0.0 - 1.0) of words to overlap between adjacent chunks (default 0.1, or 10%).

    Returns:
        List[str]: List of text chunks as strings.

    Other options:
        - Chunking by character length, sentence, or paragraph instead of word count.
        - Use recursive character/sentence chunking (e.g., LangChain’s `RecursiveCharacterTextSplitter`).
        - Maximal Marginal Relevance (MMR) chunk selection for diversity and coverage.
        - Dynamic chunking based on semantic similarity, topic shifts, or window size fit to embedding/token limits.
        - For PDFs or DOCX: run OCR/extraction, then chunk the result.
        - Use chunk labels/IDs for mapping back to original docs.
        - If supporting other languages/scripts, consider unicode-aware splitting.

    Why chunking?
        - Controls context size for retrieval-augmented generation.
        - Too large: less precise, memory-heavy. Too small: more retrieval calls, less context per answer.

    Examples:
        >>> simple_chunk_text("This is a test " * 30, chunk_size=10, overlap_pct=0.2)
        [ ...list of overlapping 10-word chunks... ]
    """
    words = text.split()
    overlap = int(chunk_size * overlap_pct)
    step = chunk_size - overlap if chunk_size > overlap else chunk_size
    chunks = []
    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(words):
            break
    return chunks
