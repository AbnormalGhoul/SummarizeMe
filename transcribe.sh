#!/bin/bash
# SBATCH --job-name=summarize_training
# SBATCH --output=training_output.log
# SBATCH --error=training_error.log
# SBATCH --time=02:00:00
# SBATCH --account=warren.christian
# SBATCH --partition=kamiak
# SBATCH --gres=gpu:1
# SBATCH --mem=16G
# SBATCH --cpus-per-task=10
# SBATCH --mail-type=ALL
# SBATCH --mail-user=warren.christian@wsu.edu
# Activate virtual environment
echo "🔄 Activating virtual environment..."
source /home/warren.christian/crimsoncode/python_files/summarizeit/bin/activate

# Load FFmpeg (if available)
module load ffmpeg || echo "⚠️ FFmpeg module not found, skipping..."
/home/warren.christian/bin/ffmpeg-7.0.2-amd64-static/ffmpeg

# Use system Python 3
PYTHON_EXEC=$(which python3)
echo "✅ Python version: $($PYTHON_EXEC --version)"

# Get Whisper executable after venv activation
WHISPER_EXEC=$(which whisper)
echo "✅ Whisper path: $WHISPER_EXEC"

AUDIO_PATH=$1
WAV_PATH="${AUDIO_PATH%.m4a}.wav"
OUTPUT_DIR="/home/warren.christian/crimsoncode/results"
mkdir -p "$OUTPUT_DIR"

# Ensure correct file extension handling
WAV_PATH="${AUDIO_PATH%.*}.wav"

echo "🔄 Converting M4A to WAV..."
/home/warren.christian/bin/ffmpeg-7.0.2-amd64-static/ffmpeg -i "$AUDIO_PATH" -ar 16000 -ac 1 -c:a pcm_s16le "$WAV_PATH"

# Run Whisper with the correct path
echo "🔄 Running Whisper Transcription..."
$WHISPER_EXEC "$WAV_PATH" --model base --output_dir "/home/warren.christian/crimsoncode/"

# Move transcription output to results folder
TEXT_FILE=$(ls "$OUTPUT_DIR" | grep "$(basename "$WAV_PATH" .wav)" | grep ".txt")
mv "$OUTPUT_DIR/$TEXT_FILE" "$OUTPUT_DIR/$(basename "$AUDIO_PATH" .m4a).txt"

# Run the Python transcription and summarization script
 # Run the Python transcription and summarization script
echo ":arrows_counterclockwise: Running Python transcription and summarization..."
python /home/warren.christian/crimsoncode/python_files/main.py "$WAV_PATH"