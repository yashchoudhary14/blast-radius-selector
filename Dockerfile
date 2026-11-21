FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS source
WORKDIR /app
COPY requirements.lock .
RUN split -n r/4 requirements.lock /tmp/lock- && \
    find /tmp -maxdepth 1 -name 'lock-*' -print0 | xargs -0 -n 1 -P 4 \
        pip download --disable-pip-version-check --no-cache-dir --no-deps -d /tmp/wheels -r && \
    PIP_NO_INDEX=1 PIP_FIND_LINKS=/tmp/wheels pip install --no-cache-dir -r requirements.lock && \
    rm -r /tmp/wheels /tmp/lock-*
RUN python - <<'PY'
import tarfile
import urllib.request
from pathlib import Path
commit = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
with urllib.request.urlopen(f"https://codeload.github.com/tobymao/sqlglot/tar.gz/{commit}", timeout=60) as response:
    with tarfile.open(fileobj=response, mode="r|gz") as archive:
        archive.extractall("/tmp", filter="data")
Path(f"/tmp/sqlglot-{commit}").rename("/sqlglot")
PY
RUN python -m compileall -q /sqlglot
ENV PYTHONPATH=/app:/sqlglot SKIP_INTEGRATION=0 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1

FROM source AS static
COPY maps/ maps/
COPY runner/ runner/
COPY selector/ selector/
RUN python -m selector.collect && python -m selector.build_maps && \
    python -m compileall -q /app && chmod -R a+rX /sqlglot /app
ENV SELECTOR_STATIC=1
CMD ["python", "-m", "selector.main"]

FROM static
# The two selector lanes share two CPUs. A CPU per default-size Python
# process pool avoids starting a second optimizer worker in each lane.
ENV SELECTOR_STATIC=0 PYTHON_CPU_COUNT=1
