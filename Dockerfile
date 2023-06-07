FROM debian:bullseye-slim as base

COPY requirements.txt /var/app/requirements.txt

RUN apt-get update -y \
    && apt-get install --no-install-recommends -y \
    python3.9 \
    python3-pip \
    git \
    && cd /var/app \
    && python3.9 -m pip install -r requirements.txt \
    && rm -rf /var/cache/apt/archives

COPY ["main.py" ,"pdf_uploader.py" ,"/var/app/"]

WORKDIR /var/app
ENTRYPOINT [ "python3", "/var/app/main.py" ]

FROM base as compressor

RUN apt-get update -y \
    && apt-get install --no-install-recommends -y \
    python3.9-dev \
    build-essential \
    ccache \
    clang \
    patchelf \
    upx \
    && python3.9 -m pip install nuitka

RUN python3.9 -m nuitka \
    --standalone \
    --static-libpython=yes \
    --nofollow-import-to=pytest \
    --nofollow-import-to=unittest \
    --show-progress \
    --python-flag=nosite,-O \
    --plugin-enable=anti-bloat,implicit-imports,data-files,pylint-warnings \
    --clang \
    --warn-implicit-exceptions \
    --warn-unusual-code \
    --prefer-source-code \
    main.py \
    && cd main.dist/ \
    && ldd main.bin | grep "=> /" | awk '{print $3}' | xargs -I '{}' cp --no-clobber -v '{}' . \
    && ldd main.bin | grep "/lib64/ld-linux-x86-64" | awk '{print $1}' | xargs -I '{}' cp --parents -v '{}' . \
    && cp --no-clobber -v /lib/x86_64-linux-gnu/libgcc_s.so.1 . \
    && mkdir -p ./lib/x86_64-linux-gnu/ \
    && cp --no-clobber -v /lib/x86_64-linux-gnu/libresolv* ./lib/x86_64-linux-gnu \
    && cp --no-clobber -v /lib/x86_64-linux-gnu/libnss_dns* ./lib/x86_64-linux-gnu \
    && upx -9 main.bin


FROM scratch

COPY --from=compressor /var/app/main.dist/ /

ENTRYPOINT [ "/main.bin" ]