# Base Image
FROM python:3.10

# Update pip
RUN python -m pip install --upgrade pip

# if not present create directory and navigate to this work directory
WORKDIR /HousePricePrediction

#Copy all files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Run flask server
CMD ["python3", "-m", "src.model.serve"]