import streamlit as st
import librosa
import librosa.display
import google.generativeai as genai
import numpy as np
import pretty_midi
import soundfile as sf
import matplotlib.pyplot as plt
import tempfile
import os

# 🔑 Configure Gemini API
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "your_gemini_api_key_here")  # Secure way for deployment
genai.configure(api_key=GEMINI_API_KEY)

# 🎵 Streamlit App Title
st.set_page_config(page_title="AI BGM Generator", layout="centered")
st.title("🎶 AI-Generated BGM from Indian Classical Music")
st.markdown("Upload an Indian classical music piece and generate a new BGM using AI!")

# 📤 File Uploader for Audio
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file:
    # Save uploaded audio to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.getvalue())
        audio_path = temp_audio.name

    # 🎵 Load the audio
    y, sr = librosa.load(audio_path)

    # 🎼 Extract features
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = [np.max(pitches[:, i]) for i in range(pitches.shape[1]) if np.max(pitches[:, i]) > 0]
    avg_pitch = np.mean(pitch_values) if pitch_values else 0

    # 📊 Display musical features
    st.subheader("🎼 Music Analysis")
    st.write(f"**Tempo:** {tempo:.2f} BPM")
    st.write(f"**Average Pitch:** {avg_pitch:.2f} Hz")

    # 🎻 User input
    raga_choice = st.text_input("🎵 Enter Raga Name (e.g., Yaman, Bhairav)", "Yaman")
    mood_choice = st.selectbox("🎭 Choose Mood", ["Peaceful", "Energetic", "Sad", "Meditative", "Joyful"])

    # 🧠 Generate AI Composition
    if st.button("🎶 Generate AI-Based BGM"):
        with st.spinner("Generating music composition... 🎼"):
            prompt = f"""
            I am analyzing an Indian classical music piece with an average pitch of {avg_pitch:.2f} Hz and a tempo of {tempo:.2f} BPM.
            The user wants to generate a new background music (BGM) based on the raga {raga_choice} with a {mood_choice.lower()} mood.
            Suggest a melody structure, note sequences, and rhythmic pattern suitable for the mood and raga.
            """
            chat = genai.GenerativeModel("gemini-pro").start_chat()
            response = chat.send_message(prompt)
            ai_suggestions = response.text.strip()

            # 🧾 Display AI suggestions
            st.subheader("🎶 AI-Generated Composition")
            st.markdown(f"```\n{ai_suggestions}\n```")

            # 🎼 Generate MIDI from pitch values
            midi = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0)
            note_durations = np.linspace(0.5, 1.5, num=len(pitch_values))
            start_time = 0.0

            for pitch, duration in zip(pitch_values, note_durations):
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=int(np.clip(pitch, 21, 108)),  # Safe MIDI pitch range
                    start=start_time,
                    end=start_time + duration,
                )
                instrument.notes.append(note)
                start_time += duration

            midi.instruments.append(instrument)

            # 💾 Save MIDI and synthesize audio
            midi_path = "generated_bgm.mid"
            audio_output = "generated_bgm.wav"
            midi.write(midi_path)

            try:
                synthesized_audio = midi.fluidsynth(fs=sr)  # Use pyfluidsynth backend if installed
            except:
                synthesized_audio = midi.synthesize(fs=sr)  # Fallback

            sf.write(audio_output, synthesized_audio, sr)
            st.success("✅ New BGM Generated Successfully!")

            # 📥 Download buttons
            with open(midi_path, "rb") as f:
                st.download_button("🎼 Download MIDI", f, file_name="generated_bgm.mid", mime="audio/midi")
            with open(audio_output, "rb") as f:
                st.download_button("🔊 Download Audio (WAV)", f, file_name="generated_bgm.wav", mime="audio/wav")

            # 🎧 Audio player
            st.audio(audio_output, format="audio/wav")

            # 📈 Pitch contour plot
            st.subheader("📈 Pitch Contour")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(pitch_values, color="purple")
            ax.set_title(f"Pitch Contour for {raga_choice} BGM")
            ax.set_xlabel("Time")
            ax.set_ylabel("Frequency (Hz)")
            st.pyplot(fig)

    # 🧹 Clean up temp file
    os.remove(audio_path)
