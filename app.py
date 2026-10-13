import gradio as gr
import joblib
import pandas as pd

from pathlib import Path


mappe = Path(__file__).resolve().parent

modell = joblib.load(mappe / "phishing_modell.pkl")


def sjekk_epost(avsender, emne, epost_tekst):

    avsender = avsender or ""
    emne = emne or ""
    epost_tekst = epost_tekst or ""

    if not any([
        avsender.strip(),
        emne.strip(),
        epost_tekst.strip()
    ]):
        return {
            "Legitim e-post": 0.0,
            "Spam/Phishing": 0.0
        }, "Venter på input..."

    input_data = pd.DataFrame({
        "sender": [avsender],
        "text": [emne + " " + epost_tekst]
    })

    sannsynligheter = modell.predict_proba(input_data)[0]

    safe_index = list(modell.classes_).index(0)
    spam_index = list(modell.classes_).index(1)

    safe_prob = float(sannsynligheter[safe_index])
    spam_prob = float(sannsynligheter[spam_index])

    resultat = {
        "Legitim e-post": safe_prob,
        "Spam/Phishing": spam_prob
    }

    prosent = spam_prob * 100

    if prosent >= 75:
        beskjed = (
            f"HØY RISIKO ({prosent:.1f}%)\n"
            "E-posten har mange kjennetegn på spam eller phishing."
        )

    elif prosent >= 40:
        beskjed = (
            f"MULIG RISIKO ({prosent:.1f}%)\n"
            "E-posten har enkelte mistenkelige kjennetegn."
        )

    else:
        beskjed = (
            f"LAV RISIKO ({prosent:.1f}%)\n"
            "Modellen finner få tegn på spam eller phishing."
        )

    return resultat, beskjed


grensesnitt = gr.Interface(
    fn=sjekk_epost,

    inputs=[
        gr.Textbox(
            lines=1,
            placeholder="support@example.com",
            label="Avsender"
        ),

        gr.Textbox(
            lines=1,
            placeholder="Email subject",
            label="Emne"
        ),

        gr.Textbox(
            lines=10,
            placeholder="Lim inn e-posten her...",
            label="E-postens innhold"
        )
    ],

    outputs=[
        gr.Label(
            label="Modellens sannsynligheter"
        ),

        gr.Textbox(
            label="Vurdering",
            lines=3
        )
    ],

    title="AI Spam & Phishing-detektor",

    description=(
        "Analyserer avsender, emne og innhold "
        "ved hjelp av maskinlæring."
    ),

    theme=gr.themes.Soft()
)


if __name__ == "__main__":
    grensesnitt.launch()