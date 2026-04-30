FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Deno (JS runtime pour yt-dlp YouTube)
RUN curl -fsSL https://deno.land/install.sh | sh
ENV DENO_INSTALL="/root/.deno"
ENV PATH="${DENO_INSTALL}/bin:${PATH}"

WORKDIR /app

# Install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# FORCE update yt-dlp (important pour YouTube)
RUN pip install --no-cache-dir -U yt-dlp

# Copy app
COPY . .

# Run server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
