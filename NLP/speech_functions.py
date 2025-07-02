import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import datetime
import os


def record_audio(duration=5, fs=44100):
    if not os.path.exists('recordings'):
        os.makedirs('recordings')
    filename = datetime.datetime.now().strftime("recordings/recording_%Y%m%d_%H%M%S.wav")
    print(f"Recording started... Saving as {filename}")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    sf.write(filename, recording, fs)  # 🔥 Correct format for speech recognition
    print(f"Recording saved as {filename}")
    return filename


def speech_to_text(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Sorry, could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"API error: {e}")
        return None
