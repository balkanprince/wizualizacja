import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Wczytanie danych
movies = pd.read_csv(r'C:\Users\mpiesio\Desktop\KODILLA\wizualizacja\tmdb_movies.csv')
genres = pd.read_csv(r'C:\Users\mpiesio\Desktop\KODILLA\wizualizacja\tmdb_genres.csv', header=None)
genres.columns = ['genre_id', 'name']

# Dodanie kolumny z rokiem
movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
movies['release_year'] = movies['release_date'].dt.year

# === ZADANIE 1 ===
# Top 10 filmów wg oceny i liczby głosów powyżej 3. kwartylu
q3_vote_count = movies['vote_count'].quantile(0.75)
top_10_movies = (
    movies[movies['vote_count'] > q3_vote_count]
    .sort_values(by='vote_average', ascending=False)
    .head(10)
    [['title', 'vote_average', 'vote_count']]
)

print("== TOP 10 NAJLEPIEJ OCENIANYCH FILMÓW (głosy > 3. kwartyl) ==")
print(top_10_movies)

# === ZADANIE 2 ===
# Średni przychód i budżet w latach 2010–2016
movies_filtered = movies[(movies['release_year'] >= 2010) & (movies['release_year'] <= 2016)]
grouped = movies_filtered.groupby('release_year')[['revenue', 'budget']].mean()

# Wykres
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(grouped.index, grouped['revenue'], label='Średni przychód', color='skyblue')
ax1.set_ylabel('USD')
ax1.set_title('Średni przychód i budżet filmów (2010–2016)')
ax1.set_xlabel('Rok')

ax2 = ax1.twinx()
ax2.plot(grouped.index, grouped['budget'], label='Średni budżet', color='red', marker='o')

lines_labels = [*ax1.get_legend_handles_labels(), *ax2.get_legend_handles_labels()]
lines, labels = lines_labels[0] + lines_labels[2], lines_labels[1] + lines_labels[3]
ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

# === ZADANIE 3 ===
# Połączenie filmów z nazwami gatunków
movies = movies.merge(genres, how='left', left_on='genre_id', right_on='genre_id')

# === ZADANIE 4 ===
# Najczęstszy gatunek
most_common_genre = movies['name'].value_counts().idxmax()
most_common_genre_count = movies['name'].value_counts().max()
print(f"\n== NAJCZĘSTSZY GATUNEK: {most_common_genre} ({most_common_genre_count} filmów) ==")

# === ZADANIE 5 ===
# Gatunek z najdłuższym średnim czasem trwania
longest_runtime_genre = movies.groupby('name')['runtime'].mean().idxmax()
longest_runtime_value = movies.groupby('name')['runtime'].mean().max()
print(f"== NAJDŁUŻSZY ŚREDNI CZAS TRWANIA: {longest_runtime_genre} ({longest_runtime_value:.2f} min) ==")

# === ZADANIE 6 ===
# Histogram długości trwania dla tego gatunku
longest_runtime_movies = movies[movies['name'] == longest_runtime_genre]
plt.figure(figsize=(10, 5))
plt.hist(longest_runtime_movies['runtime'].dropna(), bins=20, color='green', edgecolor='black')
plt.title(f'Histogram czasu trwania filmów: {longest_runtime_genre}')
plt.xlabel('Czas trwania (minuty)')
plt.ylabel('Liczba filmów')
plt.grid(True)
plt.tight_layout()
plt.show()
