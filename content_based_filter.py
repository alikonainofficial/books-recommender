import os
import pandas as pd
import ast
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from dotenv import load_dotenv

# from Levenshtein import distance as levenshtein_distance
from supabase import create_client, Client
from penalties import (
    apply_penalties,
    cross_genre_penalty,
    # enforce_author_uniqueness,
    genre_alignment_score,
)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

# Create Supabase client
supabase: Client = create_client(supabase_url, supabase_key)

tfidf = TfidfVectorizer(stop_words="english")


# Step 1: Load books metadata from CSV
def load_books_metadata(csv_file_path):
    """
    Loads and processes book metadata from a CSV file, transforming string fields into usable lists
    and creating a new 'features' column for each book entry.

    Parameters:
    -----------
    csv_file_path : str
        The file path of the CSV file containing the book metadata.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the processed book metadata. The 'ai_topics' and 'ai_categories'
        columns are converted from string representations of lists to actual lists, and a new
        'features' column is created by combining the 'main_author', 'ai_description',
        'ai_categories', and 'ai_topics' columns.

    Processing Steps:
    -----------------
    - Converts 'ai_topics' and 'ai_categories' from string representations into list objects.
    - Strips extra spaces from 'ai_description', 'ai_categories', and 'ai_topics'.
    - Combines 'main_author', 'ai_description', 'ai_categories', and 'ai_topics' into a single
      'features' column for each book, ensuring no NaN values.
    """
    logger.info(f"Loading books metadata from: {csv_file_path}")
    try:
        books_metadata_df = pd.read_csv(csv_file_path)

        # Convert ai_topics and ai_categories from string representation of lists to actual lists
        books_metadata_df["ai_topics"] = books_metadata_df["ai_topics"].apply(ast.literal_eval)
        books_metadata_df["ai_categories"] = books_metadata_df["ai_categories"].apply(
            ast.literal_eval
        )

        # Create the features by joining ai_categories, ai_description, and ai_topics
        books_metadata_df["features"] = (
            books_metadata_df["main_author"].str.strip()
            + " "
            + books_metadata_df[
                "ai_description"
            ].str.strip()  # Strip ai_description of extra spaces
            + " "
            + books_metadata_df["ai_categories"].apply(
                lambda x: " ".join([cat.strip() for cat in x])
            )  # Remove extra spaces in ai_categories
            + " "
            + books_metadata_df["ai_topics"].apply(
                lambda x: " ".join([topic.strip() for topic in x])
            )  # Remove extra spaces in ai_topics
        )
        # Ensure no NaN values in the features column
        books_metadata_df["features"] = books_metadata_df["features"].fillna("")

        logger.info("Books metadata loaded and processed successfully")
        return books_metadata_df
    except Exception as e:
        logger.error(f"Error loading books metadata: {e}")
        raise


# Step 2: Fetch user data and reading progress
def fetch_user_data(email, books_metadata_df, user_id=None):
    """
    Fetches user data, reading progress, and recommendations from the database.

    Parameters:
    -----------
    email : str
        The email address used to identify the user in the onboarding database.
    books_metadata_df : pandas.DataFrame
        A DataFrame containing book metadata, including the 'id' and 'title' columns,
        used to map book titles to the reading progress data.
    user_id : str, optional
        The unique identifier of the user, used to fetch reading progress (if available).
        If None, reading progress will not be retrieved.

    Returns:
    --------
    tuple
        A tuple containing the following:
        - user (dict or None): The user's onboarding data from the database, or None if no data is found.
        - reading_progress (list of dict or None): The user's reading progress, including book titles,
          or None if no user_id is provided or no progress is found.
        - user_recommendations (list or None): A list of valid book recommendations for the user,
          or None if no recommendations exist in the user data.
    """
    logger.info(f"Fetching user data for email")
    try:
        user_data = supabase.table("onboarding").select("*").eq("email", email).execute()

        if not user_data or not user_data.data:
            logger.warning(f"No user data found for email: {email}")
            return None, None, None  # If no user data is found, return early

        user = user_data.data[0]
        reading_progress = None

        if user_id:
            reading_progress_data = (
                supabase.table("reading_progress").select("*").eq("user_id", user_id).execute()
            )
            if reading_progress_data and reading_progress_data.data:
                reading_progress = reading_progress_data.data
                # Create a mapping of book titles by their id from books_metadata_df
                book_titles_map = books_metadata_df.set_index("id")["title"].to_dict()
                # Assign titles to the reading progress based on the books_metadata_id
                for progress in reading_progress:
                    progress["title"] = book_titles_map.get(progress["books_metadata_id"])
        # Filter valid recommendations from the user object
        user_recommendations = [rec for rec in user.get("recommendations", []) if rec]
        logger.info(f"User data fetched successfully for email.")
        return user, reading_progress, user_recommendations
    except Exception as e:
        logger.error(f"Error fetching user data: {e}")
        raise


# Step 3: Filter books based on user preferences
def filter_books(user_data, books_metadata_df, reading_progress):
    """
    Filters a list of books based on user preferences, genre, and reading history.

    Parameters:
    -----------
    user_data : dict
        A dictionary containing user preferences, including "genres" (list of genres)
        and "book_you_read" (list of titles the user has read).
    books_metadata_df : pandas.DataFrame
        A DataFrame containing book metadata with at least "title" and "ai_categories" columns.
    reading_progress : list of dict
        A list of dictionaries, each containing information about books the user has recently read,
        where each dict has a "title" key for the book's title.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame of books filtered by the user's preferred genres (if specified) and
        excluding books that the user has already read.
    """
    logger.info("Filtering books based on user preferences and reading history")
    try:
        genres_set = set(user_data.get("genres", []))
        # Concatenate books from reading_progress and book_you_read if both are available
        recently_read_books = (
            [item["title"] for item in reading_progress] if reading_progress else []
        )
        previously_read_books = user_data.get("book_you_read", [])

        # Combine both lists of books
        combined_books_read = set(
            recently_read_books + previously_read_books
        )  # Use set to avoid duplicates

        # Filter books based on genres and exclude already read books
        def has_matching_genre(ai_categories, genres_set):
            if not ai_categories:  # Handle missing or empty ai_categories
                return False
            if "Summary" in ai_categories:
                return True
            # Check if any genre from user data is in the book's ai_categories
            return not genres_set.isdisjoint(ai_categories)  # Faster set intersection

        # Apply genre filtering if genres are provided
        if genres_set:
            filtered_books = books_metadata_df[
                books_metadata_df["ai_categories"].apply(
                    lambda x: has_matching_genre(x, genres_set)
                )
            ]
        else:
            filtered_books = books_metadata_df  # No genre filter applied if genres are empty

        # Exclude books already read
        filtered_books = filtered_books[~filtered_books["title"].isin(combined_books_read)]
        logger.info(f"{len(filtered_books)} books filtered successfully based on user preferences")
        return filtered_books.reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error filtering books: {e}")
        raise


# Step 4: Create user profile based on reading history and preferences
def create_user_profile(
    user_data, books_metadata_df, reading_progress, user_recommendations
) -> str:
    """
    Creates a personalized user profile based on the user's reading progress,
    previously read books, recommendations, and preferred genres.

    Args:
        user_data (Dict[str, any]): User data containing 'book_you_read' and 'genres' information.
        books_metadata_df (pd.DataFrame): Dataframe containing book metadata, including 'id', 'title', 'features', and 'ai_categories'.
        reading_progress (Optional[List[Dict[str, any]]]): List of dictionaries representing the user's reading progress, each containing a 'title'.
        user_recommendations (List[int]): List of recommended book IDs.

    Returns:
        str: A concatenated string of book features that form the user's profile.
    """
    logger.info("Creating user profile based on reading history and preferences")
    try:
        # Concatenate books from reading_progress and book_you_read if both are available
        recently_read_books = (
            [item["title"] for item in reading_progress] if reading_progress else []
        )
        previously_read_books = user_data.get("book_you_read", [])

        # Fetch book titles based on recommendations
        if not books_metadata_df.index.isin(user_recommendations).any():
            books_metadata_df.set_index("id", inplace=True, drop=False)

        recommendation_titles = books_metadata_df.loc[user_recommendations, "title"].tolist()

        # Fetch book titles for recommendations
        recommendation_titles = books_metadata_df[
            books_metadata_df["id"].isin(user_recommendations)
        ]["title"].tolist()

        # Combine both lists of books
        combined_books_read = list(
            set(recently_read_books + previously_read_books + recommendation_titles)
        )  # Use set to avoid duplicates

        # If the user has no reading progress, base profile on genres alone
        if not reading_progress:
            genres = user_data["genres"]
            if genres:
                filtered_books = books_metadata_df[
                    books_metadata_df["ai_categories"].str.contains("|".join(genres), regex=True)
                ]
                user_profile = " ".join(filtered_books["features"].tolist())
            else:
                user_profile = ""
        else:
            # Build profile from features of combined books
            read_books_features = books_metadata_df[
                books_metadata_df["title"].isin(combined_books_read)
            ]["features"]
            user_profile = " ".join(read_books_features.tolist())

        logger.info("User profile created successfully")
        return user_profile
    except Exception as e:
        logger.error(f"Error creating user profile: {e}")
        raise


# Step 5: Calculate genre weightage
def calculate_genre_weightage(user_data, books_metadata_df, reading_progress):
    """
    Calculates the genre weightage for a user based on their reading history.

    Args:
        user_data (Dict[str, Any]): A dictionary containing user-related information, including previously read books.
        books_metadata_df (pd.DataFrame): A pandas DataFrame containing metadata of books, including the genre information under 'ai_categories'.
        reading_progress (List[Dict[str, str]]): A list of dictionaries representing the user's current reading progress, where each dictionary contains book titles.

    Returns:
        Dict[str, float]: A dictionary where the keys are genres and the values are the weightage (proportion) of each genre based on the user's reading history.
    """
    logger.info("Calculating genre weightage based on user reading history")
    try:

        # Concatenate books from reading_progress and book_you_read if both are available
        recently_read_books = (
            [item["title"] for item in reading_progress] if reading_progress else []
        )
        previously_read_books = user_data.get("book_you_read", [])

        # Combine both lists of books
        combined_books_read = list(
            set(recently_read_books + previously_read_books)
        )  # Use set to avoid duplicates

        read_books_genres = books_metadata_df[books_metadata_df["title"].isin(combined_books_read)][
            "ai_categories"
        ]

        # Flattening the genre lists and counting occurrences of each genre
        genre_count = Counter(
            genre
            for sublist in read_books_genres
            for genre in sublist  # Assuming genres are stored as strings of lists
        )

        total_books_read = len(combined_books_read)
        genre_weightage = {genre: count / total_books_read for genre, count in genre_count.items()}
        logger.info(f"Genre weightage calculated successfully: {genre_weightage}")
        return genre_weightage
    except Exception as e:
        logger.error(f"Error calculating genre weightage: {e}")
        raise


# Step 6: Topic Modeling for deeper analysis
def extract_topics(books_metadata_df, num_topics=5) -> pd.DataFrame:
    """
    Extracts topics from book descriptions using Latent Dirichlet Allocation (LDA).

    This function takes a DataFrame containing book descriptions (in the "ai_description" column)
    and performs topic modeling using LDA. It adds the topic distribution for each book as new
    columns to the DataFrame, with one column per topic. Optionally, it also adds the dominant
    topic (i.e., the topic with the highest probability for each book).

    Parameters:
    -----------
    books_metadata_df : pd.DataFrame
        A pandas DataFrame that contains a column 'ai_description' with text descriptions of books.
    num_topics : int, default=5
        The number of topics to identify in the descriptions.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with the original book metadata, augmented with additional columns for topic
        distributions (`topic_0`, `topic_1`, ..., `topic_{num_topics-1}`) and a column for the
        dominant topic (`dominant_topic`).

    Example:
    --------
    >>> books_metadata_df = pd.DataFrame({"ai_description": ["A tale of two cities", "An epic adventure"]})
    >>> extract_topics(books_metadata_df, num_topics=3)
    """
    logger.info(f"Extracting topics from book descriptions using LDA with {num_topics} topics")

    try:
        # Step 1: TF-IDF Vectorization
        tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
        count_data = tfidf_vectorizer.fit_transform(books_metadata_df["ai_description"])

        # Step 2: LDA Topic Modeling
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            learning_method="batch",
            max_iter=10,
            n_jobs=-1,
        )
        lda_topics = lda.fit_transform(count_data)

        # Step 3: Add topic distribution to DataFrame
        topic_df = pd.DataFrame(lda_topics, columns=[f"topic_{i}" for i in range(num_topics)])

        # Step 4: Optionally add dominant topic
        books_metadata_df["dominant_topic"] = lda_topics.argmax(axis=1)
        logger.info("Topics extracted and added to books metadata")
        return pd.concat([books_metadata_df, topic_df], axis=1)
    except Exception as e:
        logger.error(f"Error extracting topics: {e}")
        raise


global_recommended_books = set()


# Step 8: Recommendation logic with improved diversity and popularity weighting
def recommend_books(
    cosine_sim,
    filtered_books_df,
    genre_weightage,
    recommended_books_across_genres=None,
    genre_filter=None,
    popularity_weight=0.3,
    num_recommendations=100,
) -> list:
    """
    Recommends books based on cosine similarity scores and other factors such as genre alignment, cross-genre penalties,
    and author uniqueness.

    Args:
        cosine_sim (list): List of cosine similarity scores.
        filtered_books_df (DataFrame): DataFrame containing filtered books with metadata.
        genre_weightage (float): The weight given to the genre alignment score.
        recommended_books_across_genres (list, optional): Books already recommended across different genres.
        genre_filter (str, optional): The genre to filter books by.
        popularity_weight (float): The weight given to the book's popularity score (default: 0.3).
        num_recommendations (int): Number of books to recommend (default: 100).

    Returns:
        list: A list of recommended books with adjusted scores based on similarity, genre alignment, popularity,
        and penalties.
    """
    logger.info(f"Generating book recommendations for genre: {genre_filter}")
    try:
        sim_scores = list(enumerate(cosine_sim[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        recommended_books = []
        title_occurrences = Counter()

        for idx, score in sim_scores:
            book = filtered_books_df.iloc[idx]
            book_title = filtered_books_df.iloc[idx]["title"]
            book_author = filtered_books_df.iloc[idx]["main_author"]
            book_genres = filtered_books_df.iloc[idx]["ai_categories"]
            book_popularity = filtered_books_df.iloc[idx]["adjusted_popularity"]
            book_topics = filtered_books_df.iloc[idx][
                [f"topic_{i}" for i in range(5)]
            ].tolist()  # Topic information

            if genre_filter and genre_filter not in book_genres:
                continue

            if title_occurrences[book_title] > 0 or book_title in global_recommended_books:
                continue  # Skip if title is already recommended

            # Calculate the genre alignment score
            genre_score = genre_alignment_score(book, genre_filter, genre_weightage)

            adjusted_score = (1 - popularity_weight) * score + popularity_weight * book_popularity
            adjusted_score *= genre_score  # Apply the genre alignment score

            new_book = {
                "book_id": book["id"],
                "title": book_title,
                "main_author": book_author,
                "adjusted_score": adjusted_score,
                "popularity_score": book_popularity,
                "ai_categories": book_genres,
                "topics": book_topics,
            }

            # Apply penalties
            new_book["adjusted_score"] *= apply_penalties(new_book, recommended_books)

            # Apply the cross-genre penalty if the book appears in multiple genres
            if recommended_books_across_genres is not None:
                new_book["adjusted_score"] *= cross_genre_penalty(
                    new_book, recommended_books_across_genres
                )

            recommended_books.append(new_book)
            title_occurrences[book_title] += 1
            global_recommended_books.add(book_title)

            if len(recommended_books) >= num_recommendations:
                break

            # If there are not enough recommendations, fill with the most popular books in the genre
            if len(recommended_books) < num_recommendations:
                remaining_books = filtered_books_df[
                    filtered_books_df["ai_categories"].apply(lambda x: genre_filter in x)
                    & (
                        ~filtered_books_df["title"].isin(global_recommended_books)
                    )  # Exclude already recommended books
                ].nlargest(num_recommendations - len(recommended_books), "adjusted_popularity")

                remaining_books_dict = remaining_books.to_dict("records")

                # Ensure all remaining books have an 'adjusted_score'
                for book in remaining_books_dict:
                    # Ensure the title is not already in the recommended books (uniqueness check)
                    if title_occurrences[book["title"]] == 0:
                        if "adjusted_score" not in book:
                            # Assign adjusted_score based on adjusted_popularity as a fallback
                            book["adjusted_score"] = book.get("adjusted_popularity", 0)
                        if "book_id" not in book:
                            book["book_id"] = book.get("id", None)
                        recommended_books.append(book)
                        title_occurrences[book["title"]] += 1
                        global_recommended_books.add(book["title"])
                    if len(recommended_books) >= num_recommendations:
                        break

        # Ensure author uniqueness in the recommended books for the genre
        # recommended_books = enforce_author_uniqueness(recommended_books)
        recommended_books = sorted(
            recommended_books, key=lambda x: x["adjusted_score"], reverse=True
        )
        logger.info(f"Recommended {len(recommended_books)} books for genre: {genre_filter}")
        return recommended_books
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise


# Step 9: Insert recommendations efficiently using batch inserts
def insert_recommendations(user_id, genre, recommended_books):
    """
    Inserts personalized book recommendations for a user into the 'personalized_book_lists' table,
    based on a specified genre and list of recommended books. If there are existing recommendations
    for the user and genre, they are first removed before inserting new ones.

    Args:
        user_id (int): The ID of the user for whom the recommendations are being created.
        genre (str): The genre name of the book list to which the recommendations belong.
        recommended_books (list of dict): A list of dictionaries where each dictionary contains the book's data,
                                          particularly 'book_id' as a key.

    Raises:
        ValueError: If no genre with the given name is found or if 'recommended_books' is not in the expected format.
        Exception: If any database operation fails.

    Example:
        recommended_books = [{'book_id': 101}, {'book_id': 102}]
        insert_recommendations(1, 'Science Fiction', recommended_books)

    """
    logger.info(f"Inserting {len(recommended_books)} recommendations for user in genre '{genre}'")

    try:
        # Validate input
        if not isinstance(recommended_books, list) or not all(
            "book_id" in book for book in recommended_books
        ):
            logger.warning(
                "Invalid recommended_books format. Expected a list of dictionaries with 'book_id'."
            )
            return

        # Query to get the genre ID from the name
        genre_row = supabase.table("book_lists").select("id").eq("name", genre).execute()

        if not genre_row or not genre_row.data:
            logger.warning(f"Skipping genre '{genre}' as it was not found.")
            return

        book_list_id = genre_row.data[0]["id"]

        # Check if there are existing recommendations for this user and genre
        existing_recommendations = (
            supabase.table("personalized_book_lists")
            .select("id")
            .eq("user_id", user_id)
            .eq("book_list_id", book_list_id)
            .execute()
        )

        # If there are existing recommendations, delete them
        if existing_recommendations and existing_recommendations.data:
            recommendation_ids = [rec["id"] for rec in existing_recommendations.data]
            supabase.table("personalized_book_lists").delete().in_(
                "id", recommendation_ids
            ).execute()
            logger.info(f"Deleted {len(recommendation_ids)} existing recommendations")

        # Prepare the recommendation data for insertion
        recommendation_data = [
            {
                "book_list_id": book_list_id,
                "book_id": book["book_id"],
                "user_id": user_id,
                "position": idx + 1,
            }
            for idx, book in enumerate(recommended_books)
        ]
        # Insert the recommendation data into the 'personalized_book_lists' table
        supabase.table("personalized_book_lists").insert(recommendation_data).execute()
        logger.info(f"Successfully inserted recommendations for user {user_id} in genre '{genre}'")
    except Exception as e:
        print(f"Error inserting recommendations: {str(e)}")
        raise


# Step 0: Main process flow
def main(user_email, books_csv_path, user_id=None, num_recommendations=20):
    logger.info(f"Starting book recommendation process for user")
    try:
        # Load books metadata from CSV
        books_metadata_df = load_books_metadata(books_csv_path)
        books_metadata_df = extract_topics(books_metadata_df)

        # Fetch user data and reading progress
        user_data, reading_progress, user_recommendations = fetch_user_data(
            user_email, books_metadata_df, user_id
        )

        if not user_data:
            print(f"No data found for user: {user_email}")
            return

        # Calculate genre weightage
        genre_weightage = calculate_genre_weightage(user_data, books_metadata_df, reading_progress)

        # Filter books based on user preferences
        filtered_books_df = filter_books(user_data, books_metadata_df, reading_progress)

        # Create user profile based on reading history and/or preferences
        user_profile = create_user_profile(
            user_data, books_metadata_df, reading_progress, user_recommendations
        )

        # Compute cosine similarity
        tfidf_matrix = tfidf.fit_transform(filtered_books_df["features"])
        user_profile_vector = tfidf.transform([user_profile])
        cosine_sim = cosine_similarity(user_profile_vector, tfidf_matrix)

        # Adjust popularity
        filtered_books_df["adjusted_popularity"] = filtered_books_df["popularity_score"].apply(
            np.log1p
        )

        recommended_books_across_genres = None

        # Iterate over each genre the user is interested in
        for genre in user_data["genres"]:
            recommended_books = recommend_books(
                cosine_sim,
                filtered_books_df,
                genre_weightage.get(genre, 0.9),
                recommended_books_across_genres,
                genre_filter=genre,
                popularity_weight=0.8,
                num_recommendations=num_recommendations,
            )
            logger.info(f"Recommendations for {genre}: {len(recommended_books)} books generated")

            # for book in recommended_books:
            #     print(book["title"])

            # Update the cross-genre recommendation tracking
            if recommended_books_across_genres is None:
                recommended_books_across_genres = []
            recommended_books_across_genres.extend(recommended_books)

            # Insert the recommendations into the personalized_book_lists table
            insert_recommendations(user_id, genre, recommended_books)
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise


# Query the database for the required user information
def fetch_user_info():
    """
    Fetches user information from the 'users' and 'onboarding' tables in a Supabase database and performs an inner join based on matching email addresses.

    The function retrieves all users from the 'users' table, including their IDs and emails, and all emails from the 'onboarding' table. It then performs an inner join by matching the email addresses from both tables and returns a list of dictionaries, each containing the 'id' and 'email' of the users whose email appears in both the 'users' and 'onboarding' tables.

    Returns:
        list: A list of dictionaries where each dictionary contains:
            - id (str/int): The ID of the user.
            - email (str): The email of the user.

            Returns an empty list if any error occurs during the fetch process.
    """
    logger.info(f"Fetching users")
    try:

        # Fetch all users
        users_response = supabase.table("users").select("id, email").execute()

        if not users_response:
            print(f"Error fetching users: {users_response.json()}")
            return []

        users = users_response.data

        # Fetch onboarding emails
        onboarding_response = supabase.table("onboarding").select("email").execute()

        if not onboarding_response:
            print(f"Error fetching onboarding: {onboarding_response.json()}")
            return []

        onboarding_emails = [onb["email"] for onb in onboarding_response.data]

        # Perform inner join based on the email match
        joined_data = [
            {"id": user["id"], "email": user["email"]}
            for user in users
            if user["email"] in onboarding_emails
        ]

        logger.info(f"User info fetched successfully")
        return joined_data
    except Exception as e:
        logger.error(f"Error fetching user info: {e}")
        return []


def process_users():
    logger.info(f"Fetching and processing users")
    users = fetch_user_info()

    if users:
        for user in users:
            email = user["email"]
            user_id = user["id"]
            print(f"Processing user: {email}", user_id)
            if email == "test@gmail.com":
                main(email, "books_metadata.csv", user_id=user_id)


if __name__ == "__main__":
    process_users()
