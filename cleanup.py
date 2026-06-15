import pandas as pd

data = pd.read_csv('Amazon_BestSelling_Books_500.csv')

df = pd.DataFrame(data)


def clean_data(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df.rename(columns={'price_(usd)': 'price'}, inplace=True)
    df.rename(columns={'sub-genre': 'sub_genre'}, inplace=True)
    df = df.dropna(subset=['title', 'author'])
    df['rating'] = df['rating'].fillna(df['rating'].mean())
    df[['title', 'author']] = df[['title', 'author']].apply(lambda x: x.str.strip())
    return df

print(clean_data(df))
print(df.info())
