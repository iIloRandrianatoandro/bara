# app.py
import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor, VitsModel, AutoTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from io import BytesIO
from pydub import AudioSegment
import torchaudio
import scipy.io.wavfile
import numpy as np
import base64

# === Chargement des modèles ===

# Text-to-text
text_tokenizer = AutoTokenizer.from_pretrained("./model/text-to-text-model")
text_model = AutoModelForSeq2SeqLM.from_pretrained("./model/text-to-text-model")

# Voice-to-text (Officiel)
voice_to_text_mg_officiel_processor = Wav2Vec2Processor.from_pretrained("./model/voice-to-text-mg-officiel-model")
voice_to_text_mg_officiel_model = Wav2Vec2ForCTC.from_pretrained("./model/voice-to-text-mg-officiel-model")
voice_to_text_mg_officiel_model.eval()

# Voice-to-text (Bara)
voice_to_text_bara_processor = Wav2Vec2Processor.from_pretrained("./model/voice_to_text_bara_model")
voice_to_text_bara_model = Wav2Vec2ForCTC.from_pretrained("./model/voice_to_text_bara_model")
voice_to_text_bara_model.eval()

# Text-to-voice
tts_bara = VitsModel.from_pretrained("./model/text-to-voice-model")  # Adapter selon ton format de sauvegarde
tts_off = AutoTokenizer.from_pretrained("./model/text-to-voice-model")

# === Fonctions utiles ===

# Fonction pour traduire un texte avec un tag de direction (>>bara<< ou >>officiel<<)
def traduire(texte_source, direction_tag):
    input_text = direction_tag + " " + texte_source  # Ajouter tag de langue au texte
    inputs = text_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)  # Tokeniser
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Vérifie si CUDA est dispo
    text_model.to(device)  # Met le modèle sur le GPU ou CPU
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Met les entrées sur le même device
    
    translated = text_model.generate(**inputs, max_length=128)  # Génère la traduction
    texte_traduit = text_tokenizer.decode(translated[0], skip_special_tokens=True)  # Décode le résultat
    return texte_traduit


# Fonction pour convertir un fichier MP3 en signal audio WAV utilisable
def mp3_to_wav_array(uploaded_file):
    audio = AudioSegment.from_file(uploaded_file, format="mp3")  # Charger mp3
    audio = audio.set_frame_rate(16000).set_channels(1)  # Resampler à 16kHz mono
    buf = BytesIO()
    audio.export(buf, format="wav")  # Exporter en WAV dans la mémoire
    buf.seek(0)
    waveform, sr = torchaudio.load(buf)  # Charger le WAV avec torchaudio
    return waveform, sr  # Retourner le tableau d'audio et son taux d'échantillonnage

# Fonction pour transcrire un audio en texte avec un modèle Wav2Vec2
def transcribe_audio(waveform, processor, model):
    inputs = processor(waveform[0], sampling_rate=16000, return_tensors="pt").input_values  # Préparer entrée
    with torch.no_grad():
        logits = model(inputs).logits  # Obtenir les logits du modèle
    pred_ids = torch.argmax(logits, dim=-1)  # Choisir les prédictions les plus probables
    return processor.decode(pred_ids[0])  # Retourner la transcription

# Fonction pour synthétiser un texte en audio WAV avec un modèle TTS
def synthesize_audio(text, tts_model, output_name="output.wav"):
    inputs = text_tokenizer(text, return_tensors='pt')  # Tokenizer avec tenseurs PyTorch
    with torch.no_grad():
        waveform = tts_model(**inputs).waveform[0].numpy()  # Générer l'audio
    scipy.io.wavfile.write(output_name, rate=tts_model.config.sampling_rate, data=waveform)  # Sauvegarder dans un fichier
    return output_name


# Fonction pour générer un lien de téléchargement d’un fichier
def generate_download_link(filepath, filename):
    with open(filepath, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:audio/wav;base64,{b64}" download="{filename}">📥 Télécharger l’audio</a>'
    return href

# === Interface Streamlit ===
st.title("Traducteur bara ⇄ malgache officiel")

# Choix de la direction de traduction
direction = st.selectbox("Choisissez la direction de traduction", ["Bara -> Malgache Officiel", "Malgache Officiel -> Bara"])

direction_tag = ">>officiel<<" if direction == "Bara -> Malgache Officiel" else ">>bara<<"

# Type d'entrée
input_type = st.radio("Type d'entrée", ["Texte", "Audio (mp3)"])

# Type de sortie
output_type = st.radio("Type de sortie", ["Texte", "Audio"])

# Entrée utilisateur
if input_type == "Texte":
    texte_input = st.text_area("Entrez le texte à traduire")
    if st.button("Traduire"):
        texte_traduit = traduire(texte_input, direction_tag)
        if output_type == "Texte":
            st.success("Traduction :")
            st.write(texte_traduit)
        else:
            st.success("Audio généré :")
            tts_model = tts_off if direction == "Bara -> Malgache Officiel" else tts_bara
            audio_file = synthesize_audio(texte_traduit, tts_model)
            st.audio(audio_file, format='audio/wav')
            st.markdown(generate_download_link(audio_file, "traduction_audio.wav"), unsafe_allow_html=True)

else:
    mp3_file = st.file_uploader("Upload un fichier mp3", type=["mp3"])
    if mp3_file and st.button("Transcrire et traduire"):
        wav_array, sr = mp3_to_wav_array(mp3_file)
        processor = voice_to_text_bara_processor if direction == "Bara -> Malgache Officiel" else voice_to_text_mg_officiel_processor
        model = voice_to_text_bara_model if direction == "Bara -> Malgache Officiel" else voice_to_text_mg_officiel_model

        texte_transcrit = transcribe_audio(wav_array, processor, model)
        texte_traduit = traduire(texte_transcrit, direction_tag)

        st.write("Transcription :", texte_transcrit)

        if output_type == "Texte":
            st.success("Traduction :")
            st.write(texte_traduit)
        else:
            tts_model = tts_off if direction == "Bara -> Malgache Officiel" else tts_bara
            audio_file = synthesize_audio(texte_traduit, tts_model)
            st.audio(audio_file, format='audio/wav')
            st.markdown(generate_download_link(audio_file, "traduction_audio.wav"), unsafe_allow_html=True)
