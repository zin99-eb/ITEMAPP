
# -------------------------------
# Stage 1 : builder (installe deps)
# -------------------------------
FROM python:3.11-slim-bookworm AS builder

# Evite écriture .pyc et buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dossier de travail
WORKDIR /app

# Copie des deps en premier pour bénéficier du cache Docker
COPY requirements.txt ./

# Installe les dépendances Python (sans cache pip pour réduire la taille)
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code
COPY App.py ./App.py
COPY .streamlit/ ./.streamlit/

# -------------------------------
# Stage 2 : runtime (prod clean)
# -------------------------------
FROM python:3.11-slim-bookworm AS runtime

# Dossier de travail
WORKDIR /app

# Crée un user non-root (sécurité)
RUN adduser --disabled-password --gecos "" appuser \
 && chown -R appuser:appuser /app
USER appuser

# Copie les environnements du builder (packages + code)
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

# Expose le port Streamlit (par convention)
EXPOSE 8501

# Healthcheck Streamlit (endpoint officiel)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
 CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# EntryPoint Streamlit: App.py, écoute sur toutes les interfaces
ENTRYPOINT ["streamlit", "run", "App.py", "--server.port=8501", "--server.address=0.0.0.0"]

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

EXPOSE 8501

CMD ["streamlit", "run", "App.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
