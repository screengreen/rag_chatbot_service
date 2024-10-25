FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the poetry.lock and pyproject.toml files
COPY poetry.lock pyproject.toml ./

# Install Poetry
RUN pip install --upgrade pip && pip install poetry==1.8.3

# Install dependencies using Poetry
RUN poetry config virtualenvs.create false && poetry install

# Copy the rest of the application code
COPY . .

# Открываем порт 8000
EXPOSE 8501


HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run app.py when the container launches
ENTRYPOINT ["streamlit", "run", "streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
