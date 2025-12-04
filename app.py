import streamlit as st
import os
import librosa
import numpy as np
# IMPORTANT : Importe PIL comme ça
import PIL.Image 

# --- PATCH PILLOW (Juste après l'import) ---
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
# -------------------------------------------

from moviepy.editor import *
from moviepy.config import change_settings
import whisper


# Configuration pour Linux (Streamlit Cloud)
if os.name == 'posix':  # Si on est sur Linux/Mac
    change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})
else:
    # Ta config Windows actuelle (pour quand tu testes chez toi)
    # ... garde ton code windows ici si tu veux ...
    pass 


# --- INTERFACE GRAPHIQUE (STREAMLIT) ---
st.set_page_config(page_title="Suno to TikTok 🚀", layout="centered")

st.title("🎵 Suno to TikTok Generator 🚀")
st.markdown("Transforme tes MP3 en vidéos virales en 1 clic.")

# 1. Zone d'Upload
col1, col2 = st.columns(2)
with col1:
    audio_file = st.file_uploader("1. Ton fichier Audio (MP3)", type=["mp3", "wav"])
with col2:
    image_file = st.file_uploader("2. Ton Image de fond (JPG/PNG)", type=["jpg", "png", "jpeg"])

# Options
model_size = st.selectbox("Qualité des sous-titres (Whisper)", ["tiny", "small", "medium"], index=1)
add_lyrics = st.checkbox("Générer les sous-titres", value=True)

def create_text_clip_pil(text, duration, fontsize=60, font="arial.ttf"):
    # Version simplifiée PIL pour contourner ImageMagick
    W, H = 1080, 200 
    img = PIL.Image.new('RGBA', (W, H), (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(img)
    
    # Essai de chargement de police, sinon défaut
    try:
        font_obj = PIL.ImageFont.truetype("DejaVuSans-Bold.ttf", fontsize)
    except:
        font_obj = PIL.ImageFont.load_default()

    # Centrage approximatif
    # (On fait simple pour que ça marche partout)
    draw.text((100, 50), text, font=font_obj, fill='white', stroke_width=2, stroke_fill='black')
    
    return (ImageClip(np.array(img))
            .set_duration(duration)
            .set_position(('center', 1400)))

# --- LE MOTEUR (Fonctions cachées) ---
def process_video(audio_path, image_path):
    # Barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()

    SCREEN_SIZE = (1080, 1920)
    
    # 1. Analyse
    status_text.text("🎧 Analyse du rythme en cours...")
    y, sr = librosa.load(audio_path)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    progress_bar.progress(25)

    # Fonction de Zoom
    def resize_func(t):
        scale = 1.0
        if len(beat_times) > 0:
            closest_beat = beat_times[np.abs(beat_times - t).argmin()]
            dist = abs(t - closest_beat)
            if dist < 0.1:
                intensity = 1 - (dist / 0.1)
                scale = 1.0 + (0.12 * intensity)
        return scale

    # 2. Montage de base
    status_text.text("🎬 Montage de la vidéo de base...")
    audio_clip = AudioFileClip(audio_path)
    
    img_base = ImageClip(image_path).set_duration(audio_clip.duration)
    img_base = img_base.resize(height=SCREEN_SIZE[1])
    if img_base.w < SCREEN_SIZE[0]: img_base = img_base.resize(width=SCREEN_SIZE[0])
    img_animated = img_base.resize(lambda t: resize_func(t)).set_position('center')
    
    clips = [img_animated]
    progress_bar.progress(50)

    # 3. Sous-titres (Optionnel)
    if add_lyrics:
        status_text.text(f"📝 Transcription ({model_size})... Patience !")
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_path, fp16=False)
        
        for segment in result["segments"]:
            if segment.get("no_speech_prob", 0) > 0.45 or not segment["text"].strip():
                continue
            
            txt = segment["text"].strip()
            start_t = segment["start"]
            end_t = segment["end"]
            duration = end_t - start_t
            
            # --- CHOIX DE LA MÉTHODE TEXTE ---
            if os.name == 'posix':
                # Sur le Cloud (Linux), on utilise la version PIL (Sûre)
                txt_clip = create_text_clip_pil(txt, duration)
                txt_clip = txt_clip.set_start(start_t)
            else:
                # Sur ton PC (Windows), on garde la version Pro ImageMagick
                txt_clip = (TextClip(txt, fontsize=70, color='white', font='Arial-Bold', 
                                    stroke_color='black', stroke_width=3, method='caption', 
                                    size=(1080*0.8, None))
                            .set_position(('center', 1400))
                            .set_start(start_t)
                            .set_duration(duration))
            
            clips.append(txt_clip)

    
    progress_bar.progress(75)

    # 4. Rendu
    status_text.text("🚀 Encodage final (ça chauffe !)...")
    final_video = CompositeVideoClip(clips, size=SCREEN_SIZE).set_audio(audio_clip)
    
    # On écrit dans un fichier temporaire
    output_path = "output_video.mp4"
    final_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac", preset="ultrafast")
    
    progress_bar.progress(100)
    status_text.text("✅ Terminé !")
    return output_path

# --- BOUTON D'ACTION ---
if st.button("🚀 GÉNÉRER LA VIDÉO", type="primary"):
    if audio_file and image_file:
        # On sauvegarde les fichiers uploadés temporairement
        with open("temp_audio.mp3", "wb") as f:
            f.write(audio_file.getbuffer())
        with open("temp_image.jpg", "wb") as f:
            f.write(image_file.getbuffer())
            
        try:
            video_path = process_video("temp_audio.mp3", "temp_image.jpg")
            
            # Afficher la vidéo
            st.video(video_path)
            
            # Bouton de téléchargement
            with open(video_path, "rb") as file:
                st.download_button(
                    label="⬇️ Télécharger la vidéo",
                    data=file,
                    file_name="tiktok_viral.mp4",
                    mime="video/mp4"
                )
        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")
    else:
        st.warning("⚠️ Merci d'uploader un fichier Audio ET une Image.")

