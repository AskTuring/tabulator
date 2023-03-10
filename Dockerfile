FROM continuumio/miniconda3

COPY . /src
WORKDIR /src
# Copy the environment file to the container
# Create the environment
RUN conda env create -f environment.yml && conda clean -afy
SHELL ["conda", "run", "--no-capture-output", "-n", "tabulator", "/bin/bash", "-c"]

# Expose port 3000
EXPOSE 3000

# Start the Gunicorn server
COPY main.py .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tabulator", "uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "3000", ]

