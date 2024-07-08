# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

# ARG PYTHON_VERSION=3.10.12
# FROM python:${PYTHON_VERSION}-slim as base
FROM python:latest AS base


# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /falafel

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    # --home "/nonexistent" \
    --home "/home/falafel" \
    --shell "/sbin/nologin" \
    # --no-create-home \
    --uid "${UID}" \
    falafel

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

#Set up inveronment for Matplotlib
# ENV MPLCONFIGDIR=/path/to/writable/directory


# Create logs directory
RUN mkdir -p /var/falafel \
    && chown -R falafel /var/falafel

# Switch to the non-privileged user to run the application.
USER falafel


# Copy the source code into the container.
COPY config ./config
COPY --chown=falafel falafel ./falafel


# Gives access to write 

# Expose the port that the application listens on.
EXPOSE 8000

# Run the application.
# CMD ["ls", "-R"]
# CMD ["nvidia-smi"]
# CMD ["python", "--version"]
CMD ["gunicorn", "--config", "falafel/gunicorn_conf.py"]
# CMD ["gunicorn", "-c", "falafel/gunicorn_conf.py", "falafel.wsgi:create_app()"]
# CMD [ "python3", "falafel/dl_model/constants.py" ]
# CMD [ "python3", "falafel/z-archive/test.py" ]