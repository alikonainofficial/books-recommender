import logging
from collections import Counter
from Levenshtein import distance as levenshtein_distance

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Penalty calculation merged into a single function
def apply_penalties(book, recommended_books) -> float:
    """
    Apply a set of penalties based on title similarity, overlap, topic clustering,
    and author/genre diversity to adjust the recommendation score.

    Args:
        book (dict): The book being evaluated.
        recommended_books (list[dict]): The list of recommended books.

    Returns:
        float: The adjusted score (between 0.1 and 1.0).

    Note:
        Increasing the penalties will reduce the likelihood of recommending books
        that are too similar, while decreasing them will increase the chances of
        recommending similar books.
    """
    logger.info(f"Applying penalties to book: {book['title']}")
    penalty = 0  # Start with zero penalty

    penalty += 1 - title_similarity_penalty(
        book["title"], [rec["title"] for rec in recommended_books]
    )
    penalty += 1 - title_overlap_penalty(
        book["title"], [rec["title"] for rec in recommended_books]
    )
    penalty += 1 - topic_cluster_penalty(book, recommended_books)
    penalty += 1 - apply_diversity_penalty(book, recommended_books)

    # Apply the penalties to reduce score (capped to minimum of 0)
    adjusted_score = max(1 - penalty, 0.1)
    logger.debug(f"Penalty applied, resulting in adjusted score: {adjusted_score}")

    return adjusted_score


# Step 7: Title Similarity Penalty using Levenshtein Distance
def title_similarity_penalty(
    book_title, recommended_titles, penalty_factor=0.8, max_distance=2
) -> float:
    """
    Apply a penalty if the book title is highly similar to any recommended title
    using the Levenshtein distance metric.

    Args:
        book_title (str): The title of the book being evaluated.
        recommended_titles (list[str]): Titles of recommended books.
        penalty_factor (float): The penalty factor to apply for similar titles.
        max_distance (int): The maximum allowed Levenshtein distance for applying the penalty.

    Returns:
        float: The penalty factor (default 0.8) if a similar title is found, otherwise 1 (no penalty).

    Note:
        Increasing `penalty_factor` will impose a harsher penalty on titles with similar names,
        reducing their chances of recommendation. Increasing `max_distance` will make the function
        more lenient by tolerating more distant matches. Decreasing these values will have the opposite effects.
    """
    logger.debug(f"Applying title similarity penalty for: {book_title}")
    for rec_title in recommended_titles:
        if levenshtein_distance(book_title, rec_title) <= max_distance:
            return penalty_factor  # Apply penalty for highly similar titles
    return 1  # No penalty


# Step 8: Word Overlap Penalty
def title_overlap_penalty(book_title, recommended_titles, penalty_factor=0.3) -> float:
    """
    Apply a penalty if there's significant overlap in words between the book title
    and any of the recommended titles.

    Args:
        book_title (str): The title of the book being evaluated.
        recommended_titles (list[str]): Titles of recommended books.
        penalty_factor (float): The penalty factor to apply for overlapping words.

    Returns:
        float: The penalty factor (default 0.3) if overlap is found, otherwise 1 (no penalty).

    Note:
        Increasing `penalty_factor` will apply a harsher penalty to books with overlapping words in the title,
        while decreasing it will allow more overlap without penalty. Lowering this factor would make title overlap
        less significant in the recommendation process.
    """
    logger.debug(f"Applying title overlap penalty for: {book_title}")
    title_words = set(book_title.lower().split())
    for rec_title in recommended_titles:
        rec_title_words = set(rec_title.lower().split())
        overlap_count = len(title_words & rec_title_words)
        # Dynamic penalty: the more overlap, the higher the penalty
        if overlap_count > 0:
            return max(1 - (overlap_count * 0.1), penalty_factor)  # Scale with overlap
    return 1  # No penalty


# Apply a penalty based on the number of common topics shared between books
def topic_cluster_penalty(book, recommended_books, topic_penalty_factor=0.5) -> float:
    """
    Apply a penalty if the book shares more than one topic with any recommended book,
    indicating it's too similar in content.

    Args:
        book (dict): The book being evaluated.
        recommended_books (list[dict]): List of recommended books.
        topic_penalty_factor (float): The penalty factor for topic overlap.

    Returns:
        float: The penalty factor (default 0.5) if multiple topics overlap, otherwise 1.

    Note:
        Increasing `topic_penalty_factor` will enforce stricter penalties on books with overlapping topics,
        making topic diversity more important. Decreasing the value will tolerate more topic similarity in recommendations.
    """
    logger.debug(f"Applying topic cluster penalty for book: {book['title']}")
    max_penalty = 1
    book_topics = set(book.get("topics", []))

    for rec_book in recommended_books:
        common_topics = book_topics & set(rec_book.get("topics", []))
        if len(common_topics) > 1:  # Apply penalty if more than one common topic
            # Dynamic penalty: scale based on how many common topics exist
            max_penalty = min(max_penalty, topic_penalty_factor / len(common_topics))

    return max_penalty


# Apply penalties for lack of diversity (same author, same genre, etc.)
def apply_diversity_penalty(
    book, recommended_books, author_penalty=0.5, genre_penalty=0.4
) -> float:
    """
    Apply penalties to encourage diversity in recommendations based on author and genre similarities.

    Args:
        book (dict): The book being evaluated.
        recommended_books (list[dict]): List of recommended books.
        author_penalty (float): Penalty factor if the same author is found.
        genre_penalty (float): Penalty factor if the same genre is found.

    Returns:
        float: The compounded penalty factor based on author and genre overlap.

    Note:
        Increasing `author_penalty` or `genre_penalty` will enforce more diversity by penalizing
        books with the same author or genre more harshly. Decreasing these values will allow more
        overlap in author and genre without heavy penalties.
    """
    logger.debug(f"Applying diversity penalty for book: {book['title']}")
    penalty = 1
    # Penalty for same author
    if any(
        book["main_author"] == rec_book["main_author"] for rec_book in recommended_books
    ):
        penalty *= author_penalty

    # Dynamic genre penalty: apply based on partial genre overlap (categories list)
    for rec_book in recommended_books:
        book_genres = set(book.get("ai_categories", []))
        rec_genres = set(rec_book.get("ai_categories", []))
        overlap_genres = book_genres & rec_genres

        if overlap_genres:
            penalty *= max(
                1 - (0.1 * len(overlap_genres)), genre_penalty
            )  # Scale based on overlap

    return penalty


# Genre Alignment Scoring: Prioritizes books aligned with the specified genre filter
def genre_alignment_score(book, genre_filter, genre_weight=0.7) -> float:
    """
    Calculates the genre alignment score for a book based on the given genre filter.

    Args:
        book (dict): The book object with metadata, including categories.
        genre_filter (str): The genre to prioritize.
        genre_weight (float): The weight assigned to the genre alignment score (default: 0.7).

    Returns:
        float: A score that boosts books aligned with the genre filter, or a small penalty if not.

    Note:
        Increasing `genre_weight` will prioritize genre alignment more strongly, giving a higher score
        to books in the target genre. Lowering this value will reduce the impact of the genre filter
        on the final score.
    """
    logger.debug(f"Calculating genre alignment score for book: {book['title']}")
    return genre_weight if genre_filter in book["ai_categories"] else 0.1


def cross_genre_penalty(
    book, recommended_books_across_genres, max_occurrences=1, penalty_factor=0.9
) -> float:
    """
    Applies a penalty if the book has been recommended multiple times across genres.

    Args:
        book (dict): The current book being evaluated.
        recommended_books_across_genres (list): A list of books recommended across different genres.
        max_occurrences (int): The maximum number of allowed occurrences before applying the penalty.
        penalty_factor (float): The penalty factor to apply if occurrences exceed max_occurrences (default: 0.3).

    Returns:
        float: A penalty multiplier if the book has been recommended too many times, otherwise 1 (no penalty).

    Note:
        Increasing `max_occurrences` will allow more flexibility in recommending the same book across genres,
        while lowering `penalty_factor` will reduce the severity of the penalty. Decreasing `max_occurrences`
        or increasing `penalty_factor` will limit cross-genre repetition more strictly.
    """
    logger.debug(f"Applying cross-genre penalty for book: {book['title']}")
    occurrences = sum(
        rec["book_id"] == book["book_id"] for rec in recommended_books_across_genres
    )
    if occurrences >= max_occurrences:
        return penalty_factor  # Apply a penalty if the book has been recommended multiple times
    return 1  # No penalty if the book has not exceeded the max occurrences


# Enforce Author Uniqueness: Limits the number of books from the same author in the same genre
# Not being used as it resulted in a significant reduction in the number of recommendations
def enforce_author_uniqueness(recommended_books, max_author_occurrences=1) -> list:
    """
    Enforces author uniqueness by limiting the number of books from the same author in the recommendations.

    Args:
        recommended_books (list): List of recommended books to evaluate for uniqueness.
        max_author_occurrences (int): Maximum number of books allowed from the same author (default: 1).

    Returns:
        list: A list of unique recommended books with author limits applied.

    Note:
        Increasing `max_author_occurrences` will allow more books from the same author in the recommendations,
        while lowering this value will enforce stricter diversity among authors.
    """
    logger.info(
        f"Enforcing author uniqueness with max occurrences: {max_author_occurrences}"
    )
    unique_recommendations = []
    author_occurrences = Counter()

    for book in recommended_books:
        if author_occurrences[book["main_author"]] < max_author_occurrences:
            unique_recommendations.append(book)
            author_occurrences[book["main_author"]] += 1

    return unique_recommendations
