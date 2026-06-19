# Verified-decompilation API — pure Python + NumPy, no web framework.
# Works on any free host that runs a container (Fly.io, Render, HF Spaces, Railway).
FROM python:3.11-slim

WORKDIR /app

# deps (just numpy) — the model + server are dependency-light by design
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# the from-scratch engine the server imports.
# dataset.py defines the pickled Vocab class, and it imports lang/obfuscate/vm
# at module load — so those must be present for the checkpoint to unpickle.
COPY model.py autograd.py dataset.py lang.py obfuscate.py vm.py \
     cflang.py cfevm.py cfdataset.py ./
# the API server, the landing page, and the trained checkpoint
COPY server/app.py ./server/app.py
COPY server/static ./server/static
COPY server/model_cf.pkl ./server/model_cf.pkl

# most free hosts inject $PORT; default to 8000 locally
ENV PORT=8000
EXPOSE 8000

CMD ["python", "server/app.py"]
