FROM alpine:3.20

WORKDIR /app

RUN apk add --no-cache bash

COPY artifacts ./artifacts

ENV SPOT_SCAM_ARTIFACT_DIR=/app/artifacts

CMD ["bash"]
