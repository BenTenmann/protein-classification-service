FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ARG REQUIREMENTS=requirements.txt
ARG MODULE=protein_classification

COPY $REQUIREMENTS .
RUN pip install -r $REQUIREMENTS

WORKDIR /
COPY $MODULE $MODULE

EXPOSE 5000

ENV FLASK_APP=$MODULE/app.py
ENV FLASK_ENV=development
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
