import streamlit as st
import os
import librosa
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import gc
import tempfile
from moviepy.editor import *
from moviepy.config import change_settings
import whisper

# --- PATCH PILLOW ---
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# --- CONFIGURATION LINUX/CLOUD ---
if os.name == 'posix':
    change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})

# --- FONCTION DE CACHE (CRUCIAL POUR LA RAM) ---
@st.cache_resource
def load_whisper_model(size):
    # Charge le modèle une seule fois et le garde en mémoire
    return whisper.load_model(size)

# --- INTERFACE ---
st.set_page_config(page_title="Suno to TikTok 🚀", layout="centered")
st.title("🎵 Suno to TikTok Generator 🚀")
st.markdown("Transforme tes MP3 en vidéos virales en 1 clic.")

# 1. Upload
col1, col2 = st.columns(2)
with col1:
    audio_file = st.file_uploader("1. Ton fichier Audio (MP3)", type=["mp3", "wav"])
with col2:
    image_file = st.file_uploader("2. Ton Image de fond (JPG/PNG)", type=["jpg", "png", "jpeg"])

# Options
if os.name == 'posix':
    st.info("ℹ️ Version Cloud : Modèle 'tiny' et résolution 720p (Optimisé pour la vitesse).")
    model_size = "tiny"
    # On réduit la résolution sur le cloud pour éviter le crash RAM
    SCREEN_SIZE = (720, 1280)
else:
    model_size = st.selectbox("Qualité des sous-titres", ["tiny", "small", "medium"], index=1)
    SCREEN_SIZE = (1080, 1920)

add_lyrics = st.checkbox("Générer les sous-titres", value=True)

def create_text_clip_pil(text, duration, fontsize=50, font="arial.ttf"):
    # Adapté pour 720p (fontsize réduit)
    W, H = SCREEN_SIZE[0], 200 # Hauteur du bandeau texte
    img = PIL.Image.new('RGBA', (W, H), (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(img)
    
    try:
        font_obj = PIL.ImageFont.truetype("DejaVuSans-Bold.ttf", fontsize)
    except:
        font_obj = PIL.ImageFont.load_default()

    import textwrap
    wrapped_text = textwrap.fill(text, width=30)
    
    # Dessin centré approximatif
    draw.text((20, 20), wrapped_text, font=font_obj, fill='white', stroke_width=2, stroke_fill='black')
    
    return (ImageClip(np.array(img))
            .set_duration(duration)
            .set_position(('center', SCREEN_SIZE[1] * 0.75))) # Positionné au 3/4 bas

def process_video(audio_path, image_path):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1. Analyse
    status_text.text("🎧 Analyse du rythme en cours...")
    y, sr = librosa.load(audio_path)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    progress_bar.progress(25)

    # Fonction Zoom
    def resize_func(t):
        scale = 1.0
        if len(beat_times) > 0:
            closest_beat = beat_times[np.abs(beat_times - t).argmin()]
            dist = abs(t - closest_beat)
            if dist < 0.1:
                intensity = 1 - (dist / 0.1)
                scale = 1.0 + (0.08 * intensity) # Zoom un peu plus doux
        return scale

    # 2. Montage Base
    status_text.text("🎬 Montage de la vidéo de base...")
    audio_clip = AudioFileClip(audio_path)
    img_base = ImageClip(image_path).set_duration(audio_clip.duration)
    
    # Redimensionnement intelligent (Cover)
    img_ratio = img_base.w / img_base.h
    screen_ratio = SCREEN_SIZE[0] / SCREEN_SIZE[1]
    
    if img_ratio > screen_ratio:
        img_base = img_base.resize(height=SCREEN_SIZE[1])
    else:
        img_base = img_base.resize(width=SCREEN_SIZE[0])
        
    img_base = img_base.crop(x_center=img_base.w/2, y_center=img_base.h/2, width=SCREEN_SIZE[0], height=SCREEN_SIZE[1])
    
    img_animated = img_base.resize(lambda t: resize_func(t)).set_position('center')
    clips = [img_animated]
    progress_bar.progress(50)

    # 3. Sous-titres
    # On initialise les variables à None pour éviter le crash "undefined variable"
    model = None
    result = None
    
    if add_lyrics:
        status_text.text(f"📝 Transcription ({model_size})... Patience !")
        
        # Utilisation du CACHE ici
        model = load_whisper_model(model_size)
        result = model.transcribe(audio_path, fp16=False)
        
        for segment in result["segments"]:
            if segment.get("no_speech_prob", 0) > 0.45 or not segment["text"].strip():
                continue
            
            txt = segment["text"].strip()
            start_t = segment["start"]
            end_t = segment["end"]
            duration = end_t - start_t
            
            if os.name == 'posix':
                txt_clip = create_text_clip_pil(txt, duration)
                txt_clip = txt_clip.set_start(start_t)
            else:
                txt_clip = (TextClip(txt, fontsize=70, color='white', font='Arial-Bold',
                           stroke_color='black', stroke_width=3, method='caption',
                           size=(SCREEN_SIZE[0]*0.8, None))
                           .set_position(('center', 1400))
                           .set_start(start_t)
                           .set_duration(duration))
            clips.append(txt_clip)
            
    progress_bar.progress(75)

    # 4. Nettoyage Mémoire AVANT encodage
    # On ne supprime PAS 'model' car il est en cache, mais on peut nettoyer le reste
    del y
    if result: del result
    gc.collect()

    # 5. Rendu
    status_text.text("🚀 Encodage final (ça chauffe !)...")
    final_video = CompositeVideoClip(clips, size=SCREEN_SIZE).set_audio(audio_clip)
    
    # Fichier temporaire unique
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = tfile.name
    tfile.close()
    
    # Encodage optimisé pour Cloud (1 thread pour économiser la RAM)
    threads_count = 1 if os.name == 'posix' else None
    
    final_video.write_videofile(
        output_path, 
        fps=24, 
        codec="libx264", 
        audio_codec="aac", 
        preset="ultrafast",
        threads=threads_count # Important pour éviter le crash
    )
    
    progress_bar.progress(100)
    status_text.text("✅ Terminé !")
    
    return output_path

# --- BOUTON D'ACTION ---
if st.button("🚀 GÉNÉRER LA VIDÉO", type="primary"):
    if audio_file and image_file:
        # Sauvegarde temporaire sécurisée
        t_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        t_audio.write(audio_file.getbuffer())
        t_audio.close()
        
        t_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        t_image.write(image_file.getbuffer())
        t_image.close()
        
        try:
            video_path = process_video(t_audio.name, t_image.name)
            
            st.video(video_path)
            
            with open(video_path, "rb") as file:
                st.download_button(
                    label="⬇️ Télécharger la vidéo",
                    data=file,
                    file_name="tiktok_viral.mp4",
                    mime="video/mp4"
                )
                
            # Nettoyage final
            os.unlink(t_audio.name)
            os.unlink(t_image.name)
            # On ne supprime pas video_path tout de suite pour permettre le téléchargement
            
        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")
    else:
        st.warning("⚠️ Merci d'uploader un fichier Audio ET une Image.")
