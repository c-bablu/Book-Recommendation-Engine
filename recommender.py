import pandas as pd
from cleanup import clean_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

csv = 'Amazon_BestSelling_Books_500.csv'

def build_engine(csv):
    df = pd.read_csv(csv)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df.rename(columns={'price_(usd)': 'price'}, inplace=True)
    df = clean_data(df)
    df['tags'] = df['title'] + ' ' + df['author'] + ' ' + df['sub_genre']

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['tags'])

    similarity = cosine_similarity(tfidf_matrix)
    print(similarity.shape)

    return df, similarity

def get_recommendations(title, df, sim_matrix, top_n=6):
    titles_lower = df['title'].str.lower()
    matches = titles_lower[titles_lower == title.lower()]
    if matches.empty:
        print(f"Book '{title}' not found.")
        return None

    idx = matches.index[0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in scores if i != idx][:top_n]
    return df[['title', 'author', 'sub_genre']].iloc[top_indices].reset_index(drop=True)

if __name__ == "__main__":
    df, sim_matrix = build_engine("Amazon_BestSelling_Books_500.csv")
    print(get_recommendations("Atomic Habits", df, sim_matrix))


