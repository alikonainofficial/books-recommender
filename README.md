# Books Recommender

This repository contains a script that recommends books to users based on their reading history, preferences, and metadata of books. The system integrates with Supabase for user and book data management and uses machine learning techniques for topic modeling and recommendation generation.

## Features

	• User Profile Creation: Based on reading history, genres, and previously read books, the system creates a profile for each user.
	• Book Filtering: Filters books by genre, excluding those the user has already read.
	• Topic Modeling: Uses Latent Dirichlet Allocation (LDA) to extract topics from book descriptions.
	• Cosine Similarity for Recommendation: Uses TF-IDF and cosine similarity to recommend books aligned with user preferences.
	• Penalties and Adjustments: The recommendation logic includes penalties to improve diversity and enforce uniqueness.
	• Supabase Integration: Fetches user data and book metadata, and inserts personalized recommendations into the database.

## Requirements

The following Python libraries are required to run the script:

	• pandas
	• scikit-learn
	• numpy
	• supabase
	• python-dotenv

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

You also need to set up your environment variables for Supabase by creating a ```.env``` file:

```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

## How It Works

1. Load Book Metadata

The script loads and processes book metadata from a CSV file, converting certain fields into usable formats (e.g., lists) and creating a features column that combines important book attributes.

```load_books_metadata(csv_file_path)```

2. Fetch User Data

The system fetches user data and reading progress from the Supabase database. It retrieves user preferences, genres, and previously read books.

```fetch_user_data(email, books_metadata_df, user_id=None)```

3. Filter Books

Based on user preferences, the system filters out books the user has already read and focuses on those that match their genre preferences.

```filter_books(user_data, books_metadata_df, reading_progress)```

4. Create User Profile

Creates a user profile by combining the metadata of previously read books, genres, and recommendations.

```create_user_profile(user_data, books_metadata_df, reading_progress, user_recommendations)```

5. Topic Modeling

Performs topic extraction from book descriptions using Latent Dirichlet Allocation (LDA).

```extract_topics(books_metadata_df, num_topics=5)```

6. Generate Recommendations

The system recommends books by calculating cosine similarity between the user profile and available books, adjusting for genre alignment, popularity, and other penalties.

```recommend_books(cosine_sim, filtered_books_df, genre_weightage, ...)```

7. Insert Recommendations

After generating recommendations, the system inserts them into the Supabase database for future reference.

```insert_recommendations(user_id, genre, recommended_books)```


## Main Workflow

The script’s main function orchestrates the entire process, from loading data to generating and inserting personalized recommendations.

```main(user_email, books_csv_path, user_id=None, num_recommendations=20)```

## Usage

To run the script and process users:

    1. Ensure that your .env file is properly configured with Supabase credentials.
    2. Run the script using:
 		
				python recommend_books.py

The system will fetch users from the database, generate personalized recommendations, and insert them back into Supabase.
