import joblib
import kagglehub
import pandas as pd

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


print("Laster ned datasettet...")

path = Path(kagglehub.dataset_download(
    "naserabdullahalam/phishing-email-dataset"
))

filer = [
    "CEAS_08.csv",
    "Enron.csv",
    "Ling.csv",
    "Nazario.csv",
    "Nigerian_Fraud.csv",
    "SpamAssasin.csv"
]

data = []

for fil in filer:
    df = pd.read_csv(path / fil, on_bad_lines="skip", low_memory=False)
    df = df.reindex(columns=["sender", "subject", "body", "label"])

    df[["sender", "subject", "body"]] = (
        df[["sender", "subject", "body"]].fillna("").astype(str)
    )

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    data.append(df[df["label"].isin([0, 1])])


df = pd.concat(data, ignore_index=True)
df = df.drop_duplicates(subset=["sender", "subject", "body"])

df["text"] = df["subject"] + " " + df["body"]
df = df[
    (df["sender"].str.strip() != "") |
    (df["text"].str.strip() != "")
]

X = df[["sender", "text"]]
y = df["label"].astype(int)

print(f"\nAntall e-poster: {len(df)}")
print(f"Legitime: {(y == 0).sum()}")
print(f"Spam/Phishing: {(y == 1).sum()}")


features = ColumnTransformer([
    (
        "sender",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=20000
        ),
        "sender"
    ),
    (
        "text",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=60000
        ),
        "text"
    )
])


modell = Pipeline([
    ("features", features),
    ("classifier", LogisticRegression(max_iter=2000))
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrener modellen...")

modell.fit(X_train, y_train)

predictions = modell.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, predictions):.2%}")
print(classification_report(
    y_test,
    predictions,
    target_names=["Legitim", "Spam/Phishing"]
))


modell.fit(X, y)

joblib.dump(
    modell,
    Path(__file__).resolve().parent / "phishing_modell.pkl"
)

print("\nModell lagret som phishing_modell.pkl")