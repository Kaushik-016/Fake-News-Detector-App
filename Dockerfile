# Use Python base image
FROM python:3.11

# Set work directory
WORKDIR /app

# Copy project files
COPY . /app

# Install system dependencies for newspaper / lxml
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libxml2-dev \
    libxslt-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords wordnet

# Expose Flask port
EXPOSE 5000

# Command to run the app
CMD ["python", "app.py"]
