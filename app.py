# app.py
import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import tempfile
import os
import yt_dlp
import requests

from melodai.spectrogram import (
    CLASSES, SAMPLE_RATE, hybrid_image, segments, to_model_input,
)

# Set page config
st.set_page_config(
    page_title="MelodAI - Bollywood Music Mood Classifier",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS for dark theme styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 1rem 0;
        color: white;
        border: 2px solid #4ECDC4;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .prediction-box h3 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
    }
    .prediction-box p {
        color: white;
        margin: 0.5rem 0;
        font-size: 1.2rem;
    }
    .confidence-bar {
        height: 25px;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid white;
    }
    .confidence-breakdown {
        background: #1E1E1E;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #444;
        margin: 1rem 0;
    }
    .class-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 0.8rem 0;
        padding: 0.5rem;
        background: #2D2D2D;
        border-radius: 8px;
    }
    .class-name {
        font-weight: bold;
        font-size: 1.1rem;
        color: white;
    }
    .class-confidence {
        font-weight: bold;
        color: #4ECDC4;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
        border: 2px solid #ff9999;
    }
    .mood-description {
        background: #2D2D2D;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #4ECDC4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model('models/best_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def download_youtube_audio(youtube_url):
    """Download audio from YouTube URL"""
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'temp_audio.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
            return 'temp_audio.mp3'
    except Exception as e:
        st.error(f"Error downloading YouTube audio: {e}")
        return None

def create_spectrogram(audio_path, target_size=(224, 224)):
    """Build the model input, using the same code the training data came from.

    This function used to build its own spectrogram: a plain mel at a
    different hop length, stacked into three identical greyscale channels and
    zero-padded into the corner of a 224x224 frame, leaving 43% of the image
    black. The model had been trained on percentile-normalised three-channel
    images where red is percussive energy, green is harmonic content with
    chroma, and blue is mel with spectral contrast. It was being shown
    pictures unlike anything in its training set, and its confidences
    collapsed towards a third each.

    melodai/spectrogram.py is now the single implementation, verified to
    reproduce the training script's output exactly.
    """
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        chunks = segments(y, sr)[:4]
        return np.stack([to_model_input(hybrid_image(c, sr)) for c in chunks])
    except Exception as e:
        st.error(f"Error creating spectrogram: {e}")
        return None


def plot_spectrogram(spectrogram):
    """Plot the generated spectrogram"""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    # The real three-channel image, not one channel under a false colormap.
    # Showing channel 0 through viridis displayed the percussive band alone
    # and made it look like an ordinary mel spectrogram, which is exactly the
    # confusion that let the preprocessing drift go unnoticed.
    ax.imshow(np.clip(spectrogram, 0, 1), aspect='auto', origin='lower')
    ax.set_title('Model input: R percussive, G harmonic, B mel',
                 fontsize=12, fontweight='bold', color='white')
    ax.set_xlabel('Time', color='white')
    ax.set_ylabel('Frequency bins', color='white')
    ax.tick_params(colors='white')
    return fig

def get_mood_description(mood):
    """Get detailed description of each mood"""
    descriptions = {
        'Energetic': {
            'description': 'High-energy songs perfect for workouts, parties, or when you need a boost!',
            'characteristics': 'Fast tempo, strong beats, percussion-heavy, upbeat vocals',
            'bollywood_examples': 'Party songs, item numbers, dance tracks'
        },
        'Peaceful': {
            'description': 'Calm, soothing melodies ideal for relaxation, focus, or peaceful moments',
            'characteristics': 'Slow tempo, soft instruments, smooth vocals, minimal percussion',
            'bollywood_examples': 'Soft melodies, background scores, romantic slow songs'
        },
        'Emotional': {
            'description': 'Expressive songs that evoke deep feelings - from love to heartbreak',
            'characteristics': 'Emotional vocals, dramatic instrumentation, varying dynamics',
            'bollywood_examples': 'Sad songs, intense love ballads, emotional scenes'
        }
    }
    return descriptions.get(mood, {})

def get_song_recommendations(mood):
    """Get Bollywood song recommendations based on predicted mood"""
    recommendations = {
        'Energetic': [
            "🎵 'Senorita' (Zindagi Na Milegi Dobara)",
            "🎵 'Badtameez Dil' (Yeh Jawaani Hai Deewani)", 
            "🎵 'London Thumakda' (Queen)",
            "🎵 'Nagada Sang Dhol' (Goliyon Ki Raasleela Ram-Leela)",
            "🎵 'Ghoomar' (Padmaavat)"
        ],
        'Peaceful': [
            "🎵 'Tum Hi Ho' (Aashiqui 2)",
            "🎵 'Raabta' (Agent Vinod)",
            "🎵 'Phir Le Aya Dil' (Barfi!)",
            "🎵 'Jeene Laga Hoon' (Ramaiya Vastavaiya)",
            "🎵 'Tujhe Kitna Chahne Lage' (Kabir Singh)"
        ],
        'Emotional': [
            "🎵 'Agar Tum Saath Ho' (Tamasha)",
            "🎵 'Channa Mereya' (Ae Dil Hai Mushkil)",
            "🎵 'Teri Mitti' (Kesari)",
            "🎵 'Kalank Title Track' (Kalank)",
            "🎵 'Hamari Adhuri Kahani' (Hamari Adhuri Kahani)"
        ]
    }
    return recommendations.get(mood, [])

def main():
    # Header
    st.markdown('<h1 class="main-header">🎵 MelodAI - Bollywood Music Mood Classifier</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #888; margin-bottom: 2rem;'>
    <strong>Exclusively trained on Bollywood music!</strong> Upload a music file or YouTube URL to predict mood: 
    <strong>Energetic</strong>, <strong>Peaceful</strong>, or <strong>Emotional</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Could not load the model. Please check if 'models/best_model.h5' exists.")
        return
    
    # Input method selection
    input_method = st.radio("Choose input method:", 
                           ["Upload MP3 File", "YouTube URL"], 
                           horizontal=True)
    
    audio_file = None
    file_source = None
    
    if input_method == "Upload MP3 File":
        audio_file = st.file_uploader("Choose an MP3 file", type=['mp3', 'wav'])
        file_source = "upload"
        
    else:  # YouTube URL
        youtube_url = st.text_input("Enter YouTube URL:")
        st.warning("⚠️ Please use the **Share URL** from the 'Share' button, not your browser address bar URL")
        if youtube_url:
            if "youtube.com" in youtube_url or "youtu.be" in youtube_url:
                with st.spinner('Downloading audio from YouTube...'):
                    audio_path = download_youtube_audio(youtube_url)
                    if audio_path and os.path.exists(audio_path):
                        # Convert to file-like object for consistency
                        with open(audio_path, 'rb') as f:
                            audio_file = io.BytesIO(f.read())
                        file_source = "youtube"
                        # Clean up temp file after processing will happen later
            else:
                st.warning("Please enter a valid YouTube URL")
    
    if audio_file is not None:
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📁 Input Source")
            if file_source == "upload":
                st.audio(audio_file, format='audio/mp3')
                # File info
                file_size = len(audio_file.getvalue()) / (1024 * 1024)  # MB
                st.write(f"**File size:** {file_size:.2f} MB")
            else:
                st.success("✅ YouTube audio downloaded successfully!")
                st.audio(audio_file, format='audio/mp3')
        
        with col2:
            st.subheader("🔄 Processing")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                if file_source == "upload":
                    tmp_file.write(audio_file.getvalue())
                else:
                    tmp_file.write(audio_file.read())
                tmp_path = tmp_file.name
            
            # Create spectrogram
            with st.spinner('Creating spectrogram...'):
                spectrogram = create_spectrogram(tmp_path)
            
            if spectrogram is not None:
                # Display spectrogram
                st.subheader("📊 What the model sees")
                st.caption(
                    "Red is percussive energy, green is harmonic content and chroma, "
                    "blue is the mel spectrogram with spectral contrast. This is the "
                    "image the network classifies, not a decoration."
                )
                fig = plot_spectrogram(spectrogram[0])
                st.pyplot(fig)

                with st.spinner('Analyzing mood...'):
                    # One prediction per 45-second segment, averaged.
                    #
                    # A single window can land on an intro and miss the song.
                    # Averaging is test-time augmentation: one more forward
                    # pass, and strictly better than trusting whichever
                    # window happened to come first.
                    all_predictions = model.predict(spectrogram, verbose=0)
                    averaged = all_predictions.mean(axis=0)

                    predicted_class = int(np.argmax(averaged))
                    confidence = float(np.max(averaged))
                    predictions = [averaged]

                    class_names = CLASSES
                    class_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                    class_emojis = ['😢', '⚡', '😌']

                    if len(spectrogram) > 1:
                        st.caption(
                            f"Averaged over {len(spectrogram)} segments of 45 seconds."
                        )

                    st.subheader("🎯 Prediction Results")
                    
                    # Main prediction box
                    predicted_mood = class_names[predicted_class]
                    confidence_percent = confidence * 100
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>{class_emojis[predicted_class]} {predicted_mood}</h3>
                        <p><strong>Confidence:</strong> {confidence_percent:.1f}%</p>
                        <div class="confidence-bar" style="width: {confidence_percent}%"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add interpretation help
                    if confidence_percent < 50:
                        st.info("🤔 **Lower confidence note**: This song shares characteristics with multiple mood categories, which is common in Bollywood music that often blends emotions!")
                    else:
                        st.success("✅ **Clear prediction**: The model is confident about this mood classification!")
                    
                    # Mood Description
                    mood_info = get_mood_description(predicted_mood)
                    st.subheader("📖 Mood Description")
                    st.markdown(f"""
                    <div class="mood-description">
                        <h4>What does <strong>{predicted_mood}</strong> mean?</h4>
                        <p><strong>Description:</strong> {mood_info['description']}</p>
                        <p><strong>Characteristics:</strong> {mood_info['characteristics']}</p>
                        <p><strong>Bollywood Examples:</strong> {mood_info['bollywood_examples']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Song Recommendations
                    st.subheader("🎶 Recommended Bollywood Songs")
                    recommendations = get_song_recommendations(predicted_mood)
                    st.markdown(f"""
                    <div class="recommendation-box">
                        <h4>If you like this {predicted_mood.lower()} mood, try these:</h4>
                        {"<br>".join(f"• {song}" for song in recommendations)}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence breakdown
                    st.subheader("📈 Confidence Breakdown")
                    st.markdown('<div class="confidence-breakdown">', unsafe_allow_html=True)
                    
                    for i, (class_name, color, emoji) in enumerate(zip(class_names, class_colors, class_emojis)):
                        conf = predictions[0][i] * 100
                        
                        st.markdown(f"""
                        <div class="class-row">
                            <span class="class-name">{emoji} {class_name}</span>
                            <span class="class-confidence">{conf:.1f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Progress bar
                        st.progress(float(conf / 100))
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Clean up temp files
                os.unlink(tmp_path)
                if file_source == "youtube" and os.path.exists('temp_audio.mp3'):
                    os.unlink('temp_audio.mp3')
                
            else:
                st.error("Failed to process the audio file. Please try another file.")
    
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About MelodAI")
        st.markdown("""
        **Exclusively for Bollywood Music!**
        
        **How it works:**
        1. Upload MP3 or paste YouTube URL
        2. AI converts audio to mel-spectrogram
        3. Deep learning model analyzes patterns
        4. Predicts mood with confidence scores
        
        **Model Performance:**
        - Overall Accuracy: 66%
        - Energetic: 88% F1-score
        - Emotional: 74% F1-score
        - Peaceful: 27% F1-score
        
        **Note:** Peaceful vs Emotional can be challenging to distinguish due to acoustic similarities.
        """)
        
        st.header("🔧 Technical Details")
        st.markdown("""
        - **Framework:** TensorFlow/Keras
        - **Architecture:** Custom CNN
        - **Input:** Mel-spectrograms (224×224)
        - **Training Data:** 1,124 Bollywood spectrograms
        - **Classes:** Energetic, Peaceful, Emotional
        """)
        
        # Model info
        st.header("🤖 Model Architecture")
        st.write(f"Output shape: {model.output_shape}")
        
        # Confidence test - MOVED INSIDE THE SIDEBAR AND MAIN FUNCTION
        st.header("🎯 Confidence Test")
        if st.button("Test Confidence Ranges"):
            # Test patterns that should trigger high/low confidence
            test_cases = {
                "High Confidence (Energetic)": np.ones((224, 224, 3)) * 0.9,  # Bright spectrogram
                "Low Confidence (Mixed)": np.ones((224, 224, 3)) * 0.5,      # Medium
                "High Confidence (Peaceful)": np.ones((224, 224, 3)) * 0.1,  # Dark spectrogram
            }
            
            for name, pattern in test_cases.items():
                test_input = np.expand_dims(pattern, axis=0)
                test_pred = model.predict(test_input, verbose=0)
                max_conf = np.max(test_pred) * 100
                st.write(f"{name}: {max_conf:.1f}% max confidence")
        
        # Mood definitions
        st.header("🎭 Mood Definitions")
        st.markdown("""
        **Energetic** ⚡
        - Fast tempo, strong beats
        - Perfect for dance and parties
        - High percussion energy
        
        **Peaceful** 😌  
        - Slow, calming melodies
        - Ideal for relaxation
        - Soft instrumentation
        
        **Emotional** 😢
        - Expressive, heartfelt
        - Evokes deep feelings
        - Dramatic vocals
        """)

if __name__ == "__main__":
    main()