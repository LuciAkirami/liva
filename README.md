# LIVA (Local Intelligent Voice Assistant)

LIVA is a project aimed at creating a local intelligent voice assistant that leverages the power of large language models (LLMs) to understand and respond to user queries in natural language. This project provides a framework for building a voice-controlled interface that integrates speech recognition, natural language processing, and text-to-speech synthesis.

## Features

- Speech-to-text conversion for transcribing user input
- Interaction with large language models (LLMs) for understanding user queries
- Text-to-speech synthesis for generating responses
- Customizable settings for specifying model configurations and API endpoints

## Installation

1. Clone the repository to your local machine:

   ```
   git clone https://github.com/LuciAkirami/liva.git
   ```

2. Install the latest torch version

   ```
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

## Usage

1. Navigate to the project directory:

   ```
   cd liva
   ```

2. Run the `main.py` script with the desired command-line arguments:

   ```
   python main.py --url <Chat Endpoint URL> --model-id <LLM Model ID> --api-key <API Key> --stt-model <Speech2Text Model>
   ```

   - `--url`: Specify the URL of the OpenAI compatible chat endpoint.
   - `--model-id`: Provide the ID of the LLM model to power the text generator.
   - `--api-key`: Provide the API Key for the relevant chat endpoint URL.
   - `--stt-model`: Specify the Speech2Text model for converting speech to text.

3. Once the script is running, speak into the microphone to interact with LIVA. The assistant will transcribe your speech, process it using the specified LLM model, and generate a response.

## Example

```bash
python main.py --url http://localhost:11434/v1 --model-id mistral:instruct --api-key ollama --stt-model openai/whisper-base.en
```
The above are the default arguments when you do not specify any and just run the following
```bash
python main.py
```
## Requirements

- Python 3.10.3 (Better to use conda for managing the environment)

## Contributing

Contributions to LIVA are welcome! If you have ideas for new features, improvements, or bug fixes, feel free to open an issue or submit a pull request.

## Potential Issues and Solutions

### Whisper Issues (libcuda issue)

#### Issue:
You may encounter errors related to libcuda when running Whisper.

#### Solution:
Ensure that `libcudnn_ops_infer.so.8` is in your library path. You can export the library path using the following command:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/{user_name}/miniconda3/envs/{env_name}/lib/{python_version}/site-packages/nvidia/cudnn/lib/
```

### PyAudio Issues (Issues Installing PyAudio)

#### Issue:
You may encounter issues installing PyAudio due to missing dependencies.

#### Solution 1:
Install PortAudio and then install PyAudio using pip:

```bash
sudo apt-get install portaudio19-dev
pip install PyAudio
```

#### Solution 2:
Alternatively, you can use conda to install PyAudio, which will also install PortAudio:

```bash
conda install PyAudio
```

### Speech5 Tokenizer

#### Issue:
To use the Speech5 Tokenizer, ensure that the sentence-piece tokenizer is installed.

### ALSA Errors

#### Issue:
You may encounter ALSA errors even if you are not using the library.

#### Solution:
Import `sounddevice` library in your code, even if you are not using it. This can help resolve ALSA errors:

```python
import sounddevice
```

Ensure that your system's audio configuration is correct and that all necessary audio devices are properly configured.

## Troubleshooting

If you encounter any other issues not covered here, please refer to the documentation of the specific libraries or tools you are using. Additionally, searching online forums and communities for similar issues can often provide helpful insights and solutions. If the problem persists, feel free to open an issue on the project's GitHub repository for further assistance.

## License

This project is licensed under the MIT License

