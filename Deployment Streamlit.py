import streamlit as st
from tensorflow.keras.models import load_model # type: ignore
import librosa
import numpy as np
import soundfile as sf
from PIL import Image


model = load_model(r'C:\Users\andre\OneDrive - Asia Pacific University\FYP\FYP report\deploy\50K_best_gru_model.h5')

def preprocess_audio(file_path, max_len=100):
    audio, sample_rate = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, n_chroma=12)

    def pad_or_truncate(feature, max_len):
        if feature.shape[1] > max_len:
            feature = feature[:, :max_len]
        else:
            pad_width = max_len - feature.shape[1]
            feature = np.pad(feature, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return feature

    mfccs = pad_or_truncate(mfccs, max_len)
    mel = pad_or_truncate(mel, max_len)
    chroma = pad_or_truncate(chroma, max_len)

    features = np.concatenate((mfccs, mel, chroma), axis=0)
    features = features.T
    features = np.expand_dims(features, axis=0)

    return features

def predict_emotion(model, file_path):
    features = preprocess_audio(file_path)
    predictions = model.predict(features)
    predicted_emotion = np.argmax(predictions)

    print("Predicted emotion (integer):", predicted_emotion)

    emotion_mapping = {0: 'NEU', 1: 'HAP', 2: 'FEA', 3: 'DIS', 4: 'SAD', 5: 'ANG'}
    print("Predicted emotion:", predicted_emotion)
    return emotion_mapping[predicted_emotion]


def convert_mp3_to_wav(file_path):
    y, sr = librosa.load(file_path, sr=None)
    output_wav_path = file_path.replace('.mp3', '.wav')
    # Write the WAV file using soundfile
    sf.write(output_wav_path, y, sr)
    return output_wav_path

def get_emoji_image(emotion, fixed_size=(704,450)):
    emoji_images = {
        'NEU': 'NeutralEmoji.jpg',  
        'HAP': 'happyEmoji.jpg',
        'FEA': 'FearEmoji.jpg',
        'DIS': 'DisgustEmoji.jpg',
        'SAD': 'SadEmoji.jpg',
        'ANG': 'AngryEmoji.jpg'
    }
    image_path = emoji_images.get(emotion, 'images/NeutralEmoji.png')
    image = Image.open(image_path)
    resized_image = image.resize(fixed_size)  
    return resized_image


def main():
    st.title("Speech Emotion Recognition")
    
    # Upload box for the user to upload a file
    uploaded_file = st.file_uploader("Upload an MP3 file", type=["mp3","wav"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'mp3':
            temp_file_path = "temp.mp3"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            # Convert the MP3 file to WAV
            wav_file = convert_mp3_to_wav("temp.mp3")

        elif file_extension == 'wav':
            wav_file = "temp.wav"
            with open(wav_file, "wb") as f:
                f.write(uploaded_file.getbuffer())

        emotion = predict_emotion(model, wav_file)

        # Map the prediction to emotion labels
        emotion_labels = {0: 'NEU', 1: 'HAP', 2: 'FEA', 3: 'DIS', 4:'SAD', 5:'ANG'}
        st.write(f"The predicted emotion is: {emotion}")
        emoji_image = get_emoji_image(emotion)
        st.image(emoji_image, caption=f"Emotion: {emotion}", use_column_width=True)


if __name__ == "__main__":
    main()