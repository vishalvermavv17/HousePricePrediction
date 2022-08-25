# Base Image
FROM python:3.10

ENV PORT 5000

# Update pip
RUN python -m pip install --upgrade pip

# if not present create directory and navigate to this work directory
WORKDIR /HousePricePrediction

#Copy all files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Run flask server
EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT src.model.serve:app