FROM python:3.8-slim-bullseye

RUN apt-get update && apt-get install -y \
     texlive-latex-base \
     texlive-latex-extra \
     texlive-fonts-recommended \
     latexmk \
     && rm -rf /var/lib/apt/lists/*

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container.
COPY . .

#Create output directories
RUN mkdir -p /app/output/plots /app/output/latex

# Expose the port that the application listens on.
EXPOSE 8000

# Run the application.
CMD ["python3", "multidim_sensor_analysis.py"]
