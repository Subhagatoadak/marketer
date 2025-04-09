# Use the official Python slim base image.
FROM python:3.10-slim

# Install system dependencies required by Pillow and other packages.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory.
WORKDIR /app

# Copy the requirements.txt first to leverage Docker cache.
COPY requirements.txt .

# Upgrade pip and install Python dependencies.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code.
COPY . .

# Expose port 8501 for Streamlit.
EXPOSE 8501

# Optional: Disable CORS if needed.
ENV STREAMLIT_SERVER_ENABLECORS=false

# Start the Streamlit application.
CMD ["streamlit", "run", "app.py"]
