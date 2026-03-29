# Use Python 3.9 as base for Hugging Face Spaces
FROM python:3.9

# Create a non-root user (required by HF)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Install dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the files
COPY --chown=user . /app

# Expose ports (Gradio: 7860, FastAPI: 8000)
EXPOSE 7860 8000

# Start the FastAPI server with Gradio mounted
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]