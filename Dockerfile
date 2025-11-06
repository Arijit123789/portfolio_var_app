# Base image
FROM python:3.10-slim

# Working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0


