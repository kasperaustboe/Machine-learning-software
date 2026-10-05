import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

file_path = "TMDB_movie_dataset_v11.csv"
print("Laster ned datasettet fra Kaggle.")

hf_dataset = kagglehub.dataset_load(
  KaggleDatasetAdapter.HUGGING_FACE,
  "asaniczka/tmdb-movies-dataset-2023-930k-movies",
  file_path
)

print("Hugging Face Dataset lastet inn vellykket!")

df = hf_dataset.to_pandas()

print(f"Datasettet inneholder {len(df)} filmer!")

df = df.dropna(subset= ['title', 'overview'])

df = df[df['vote_count'] > 100]

print(f"Datasettet inneholder {len(df)} filmer med mer enn 100 stemmer og som har en beskrivelse!")

print(df.head())
