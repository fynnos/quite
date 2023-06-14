FROM mambaorg/micromamba:1.4.3-alpine as base
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && \
    micromamba clean --all --yes

FROM base as demorphy
USER root
RUN apk add build-base
COPY --chown=$MAMBA_USER:$MAMBA_USER DEMorphy /tmp/DEMorphy
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)
RUN cd /tmp/DEMorphy && python setup.py install

FROM base as prod
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)
RUN pip install --no-cache-dir iwnlp && \
    python -m spacy download de_core_news_lg && \
    python -m spacy download en_core_web_lg && \
    rm -Rf /home/mambauser/.cache/pip
COPY --from=demorphy /opt/conda/lib/python3.9/site-packages/DAWG-0.8.0-py3.9-linux-x86_64.egg /opt/conda/lib/python3.9/site-packages/DAWG-0.8.0-py3.9-linux-x86_64.egg
COPY --from=demorphy /opt/conda/lib/python3.9/site-packages/demorphy-1.0-py3.9.egg /opt/conda/lib/python3.9/site-packages/demorphy-1.0-py3.9.egg
COPY --from=demorphy /opt/conda/lib/python3.9/site-packages/easy-install.pth /opt/conda/lib/python3.9/site-packages/easy-install.pth
COPY DEMorphy/demorphy/data/words.dg /opt/conda/lib/python3.9/site-packages/demorphy-1.0-py3.9.egg/demorphy/data/words.dg
COPY quite /quite
WORKDIR /