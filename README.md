# PDF & Notes Summarizer

A small Streamlit study app that lets students upload a PDF, generate a short summary, and ask questions about the uploaded text. It supports English and Hindi answers.

## Run locally

1. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

2. Add your OpenAI API key to a local `.env` file. This file is ignored by Git and is never uploaded:

   ```text
   OPENAI_API_KEY=your_key_here
   ```

3. Start the app:

   ```powershell
   streamlit run app.py
   ```

Then open the local address shown in the terminal, usually `http://localhost:8501`.

## Notes

- Keep `.env` private. Never upload an API key to GitHub.
- `render.yaml` contains the Render web-service configuration. In Render, add `OPENAI_API_KEY` as a secret environment variable before the first deploy.
