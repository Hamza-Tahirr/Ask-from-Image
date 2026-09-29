# Ask from Image

A Streamlit app where you upload an image and ask questions about it in plain English. A LangChain agent running on OpenAI's `gpt-3.5-turbo` answers by calling two Hugging Face vision models that run locally: BLIP for image captioning and DETR for object detection.

## Features

- Upload a JPG or PNG image and preview it in the browser
- Ask free-form questions such as "What is in this picture?" or "How many people are there?"
- The agent picks the right tool for the question:
  - **Image captioner** - `Salesforce/blip-image-captioning-large`, returns a short caption
  - **Object detector** - `facebook/detr-resnet-50`, returns bounding boxes, class names and confidence scores for detections above 0.9
- Both vision models run on CPU, so no GPU is needed

## How it works

1. The uploaded image is saved to a temporary file.
2. Your question and the file path are sent to a `chat-conversational-react-description` agent.
3. The agent calls the captioning and/or object detection tool on the image.
4. The LLM reads the tool output and writes the answer, which is shown on the page.
5. The temporary file is deleted.

## Tech stack

- Python
- Streamlit
- LangChain with the OpenAI API
- Hugging Face Transformers, PyTorch, Pillow

## Project structure

```
main.py           Streamlit UI and LangChain agent setup
tools.py          LangChain tools for captioning and object detection
functions.py      BLIP captioning and DETR object detection functions
requirements.txt  Python dependencies
.env.example      Example environment file
```

## Setup

The LangChain, Streamlit and Transformers versions in `requirements.txt` are pinned to the older releases this code was written for, so a Python 3.9 to 3.11 environment is recommended.

```bash
git clone https://github.com/Hamza-Tahirr/Ask-from-Image.git
cd Ask-from-Image

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file from the example and add your OpenAI API key:

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=your-openai-api-key
```

You can also set `OPENAI_API_KEY` as a normal environment variable instead of using `.env`.

## Run

```bash
streamlit run main.py
```

Then open the local URL Streamlit prints, upload an image and type a question.

The first time each tool runs, it downloads its model weights from Hugging Face, so the first answers take a while. Later runs use the local cache. To run the captioning model on a GPU, change `device` in `functions.py` to `"cuda"`.

## Credits

This project is based on [ask-question-image-web-app-streamlit-langchain](https://github.com/computervisioneng/ask-question-image-web-app-streamlit-langchain) by Computer vision engineer, which is also MIT licensed. On top of the original code, this version reads the OpenAI key from the environment, fixes the temporary file handling for uploaded images, removes the duplicated model code in `tools.py` and lists all the required dependencies.

## License

MIT, see [LICENSE](LICENSE). The original author's copyright notice is kept there.
