FROM python:3.9-slim

# Install system dependencies
# git: for installing dependencies from git if any
# build-essential: for compiling C extensions
# wget: for downloading resources
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
# Using poetry
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Install spacy model
RUN python -m spacy download en_core_web_sm

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# Initialize pymystem3 (downloads mystem binary)
RUN python -c "import pymystem3; pymystem3.Mystem()"

# Set entrypoint or command
CMD ["python", "scripts/train_and_infer.py"]


