# Install required libraries
# pip install torch transformers soundfile librosa pydub sentencepiece

import torch
import os
import sys
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
import librosa  # For resampling audio

# Suppress tokenizer parallelism warnings and other logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Define the results directory
RESULTS_DIR = "/home/warren.christian/crimsoncode/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Step 1: Load Whisper Distilled for Audio Transcription
def load_whisper_model():
    processor = WhisperProcessor.from_pretrained("distil-whisper/distil-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained("distil-whisper/distil-large-v3")
    return processor, model

# Step 2: Transcribe Audio (Full Transcription)
def transcribe_audio(audio_path, processor, model):
    # Load audio file and resample to 16,000 Hz
    audio_input, sample_rate = librosa.load(audio_path, sr=16000)

    # Split audio into 30-second chunks (Whisper processes 30s at a time)
    chunk_size = 30 * sample_rate  # 30 seconds in samples
    chunks = [audio_input[i:i + chunk_size] for i in range(0, len(audio_input), chunk_size)]

    # Transcribe each chunk
    full_transcription = ""
    for chunk in chunks:
        # Preprocess audio chunk
        input_features = processor(chunk, sampling_rate=sample_rate, return_tensors="pt").input_features

        # Generate transcription for the chunk
        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        # Decode transcription for the chunk
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        full_transcription += transcription[0] + " "  # Append to the full transcription

    return full_transcription.strip()  # Remove trailing spaces

# Step 3: Summarize Transcribed Text
def summarize_text(text, max_length=130, min_length=30, num_beams=4):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=max_length, min_length=min_length, num_beams=num_beams, do_sample=False)
    return summary[0]['summary_text']

# Step 4: Generate Questions from Summarized Text
def load_question_generation_model():
    model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-small-qg-hl")
    tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-small-qg-hl")
    return model, tokenizer

def generate_questions(text, model, tokenizer, max_length=64, num_beams=4):
    input_text = "generate questions: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    question_ids = model.generate(inputs["input_ids"], max_length=max_length, num_beams=num_beams)
    questions = tokenizer.decode(question_ids[0], skip_special_tokens=True)
    return questions.split("? ")

# Step 5: Main Pipeline
def main(audio_path):
    # Get base filename (without extension) for saving results
    base_filename = os.path.splitext(os.path.basename(audio_path))[0]

    # Define output file paths
    transcription_file = os.path.join(RESULTS_DIR, f"{base_filename}_transcription.txt")
    summary_file = os.path.join(RESULTS_DIR, f"{base_filename}_summary.txt")
    questions_file = os.path.join(RESULTS_DIR, f"{base_filename}_questions.txt")

    # Load models
    whisper_processor, whisper_model = load_whisper_model()
    qg_model, qg_tokenizer = load_question_generation_model()

    # Transcribe audio
    transcription = transcribe_audio(audio_path, whisper_processor, whisper_model)
    print("✅ Full Transcription Saved!")

    # Save transcription to file
    with open(transcription_file, "w") as f:
        f.write(transcription)

    # Summarize transcription
    summary = summarize_text(transcription)
    print("✅ Summary Saved!")

    # Save summary to file
    with open(summary_file, "w") as f:
        f.write(summary)

    # Generate questions
    questions = generate_questions(summary, qg_model, qg_tokenizer)
    print("✅ Generated Questions Saved!")

    # Save questions to file
    with open(questions_file, "w") as f:
        f.write("\n".join([q + "?" for q in questions]))

    return transcription, summary, questions

# Example Usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Error: No audio file provided!")
        sys.exit(1)

    audio_path = sys.argv[1]
    transcription, summary, questions = main(audio_path)

    print(f"\n📂 Results saved in: {RESULTS_DIR}")
