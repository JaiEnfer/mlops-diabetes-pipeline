FROM python:3.11-slim

WORKDIR /app

#Install dependencies first 
COPY requirements.api.txt .
RUN pip install --no-cache-dir -r requirements.api.txt

#Copy project code
COPY . .

#Expose API Port
EXPOSE 8000

#Run API
CMD [ "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]