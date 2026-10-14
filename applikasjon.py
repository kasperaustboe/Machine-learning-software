import joblib
import kagglehub
import pandas as pd

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


print("Laster ned datasettet...")

path = Path(kagglehub.dataset_download(
    "mobeenfatimah/phishing-email-detection-dataset-50k-emails"
))

df = pd.read_csv(
    path / "phishing_email_detection_dataset_v2.csv",
    low_memory=False
)

df = df[["sender_email", "subject", "email_body", "label"]].copy()

df[["sender_email", "subject", "email_body"]] = (
    df[["sender_email", "subject", "email_body"]]
    .fillna("")
    .astype(str)
)

df["label"] = (
    df["label"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "legitimate": 0,
        "phishing": 1
    })
)

df = df.dropna(subset=["label"])
df = df.drop_duplicates(
    subset=["sender_email", "subject", "email_body"]
)

df["text"] = df["subject"] + " " + df["email_body"]
df = df[df["text"].str.strip() != ""]

X = pd.DataFrame({
    "sender": df["sender_email"],
    "text": df["text"]
})

y = df["label"].astype(int)

print(f"\nAntall e-poster: {len(df)}")
print(f"Legitime: {(y == 0).sum()}")
print(f"Phishing: {(y == 1).sum()}")


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
            max_df=0.98,
            max_features=60000,
            sublinear_tf=True
        ),
        "text"
    )
])


modell = Pipeline([
    ("features", features),
    ("classifier", LogisticRegression(
        max_iter=2000,
        solver="liblinear"
    ))
])


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


parametere = {
    "classifier__C": [0.1, 1, 5]
}

sok = GridSearchCV(
    modell,
    parametere,
    cv=3,
    scoring="f1_macro",
    n_jobs=1
)


print("\nFinner beste modell...")

sok.fit(X_train, y_train)

modell = sok.best_estimator_

print("\nBeste parametere:")
print(sok.best_params_)


def evaluer(X_test, y_test, navn):

    predictions = modell.predict(X_test)

    print(f"\n{navn}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")

    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Legitim", "Phishing"],
        zero_division=0
    ))

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    print(f"Legitime feilklassifisert: {fp}")
    print(f"Phishing ikke oppdaget: {fn}")


evaluer(X_test, y_test, "Intern test")


print("\nTrener på hele datasettet...")

modell.fit(X, y)

joblib.dump(
    modell,
    Path(__file__).resolve().parent / "phishing_modell.pkl"
)

print("\nModell lagret som phishing_modell.pkl")


print("\nLaster inn GitHub-testdata...")

url = (
    "https://raw.githubusercontent.com/"
    "SunnyThakur25/Phishing-Benign-Email-Dataset-Short-Version-/"
    "main/phishing%20and%20benign%20email%20dataset.jsonl"
)

test = pd.read_json(url, lines=True)

test["label"] = test["label"].str.lower().map({
    "benign": 0,
    "phishing": 1
})

test = test.dropna(subset=["label"])

X_ekstern = pd.DataFrame({
    "sender": test["spoofed_sender"].fillna("").astype(str),
    "text": (
        test["subject"].fillna("").astype(str)
        + " "
        + test["body"].fillna("").astype(str)
    )
})

evaluer(
    X_ekstern,
    test["label"].astype(int),
    "GitHub-test"
)


print("\nLaster inn Zenodo-testdata...")

url = (
    "https://zenodo.org/records/13474746/files/"
    "Phishing_validation_emails.csv?download=1"
)

test = pd.read_csv(url)

test = test.dropna(subset=["Email Text", "Email Type"])
test = test.drop_duplicates(subset=["Email Text"])

test["label"] = test["Email Type"].str.strip().map({
    "Safe Email": 0,
    "Phishing Email": 1
})

test = test.dropna(subset=["label"])

X_ekstern = pd.DataFrame({
    "sender": [""] * len(test),
    "text": test["Email Text"].astype(str)
})

evaluer(
    X_ekstern,
    test["label"].astype(int),
    "Zenodo-test"
)