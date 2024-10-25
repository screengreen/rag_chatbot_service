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

# Run app.py when the container launches
CMD ["python3", "main.py"]
