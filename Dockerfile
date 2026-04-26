# --- build stage: install LLVM 21 + full aarch64 toolchain ---
FROM ubuntu:24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG LLVM_VERSION=21

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg lsb-release software-properties-common \
      build-essential \
      qemu-user-static binutils-aarch64-linux-gnu gcc-aarch64-linux-gnu \
  && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
      | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
  && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/noble/ llvm-toolchain-noble-${LLVM_VERSION} main" \
      > /etc/apt/sources.list.d/llvm.list \
  && apt-get update \
  && (apt-get install -y --no-install-recommends \
        clang-${LLVM_VERSION} llvm-${LLVM_VERSION} llvm-${LLVM_VERSION}-tools lld-${LLVM_VERSION} \
      || (echo "LLVM ${LLVM_VERSION} unavailable, falling back to latest" \
          && apt-get install -y --no-install-recommends clang llvm llvm-tools lld)) \
  && rm -rf /var/lib/apt/lists/*

# --- final stage: slim Python image with only needed binaries ---
FROM python:3.11-slim

COPY --from=build /usr/bin/llvm-mca-21           /usr/local/bin/llvm-mca-21
COPY --from=build /usr/bin/llvm-mca-21           /usr/local/bin/llvm-mca
COPY --from=build /usr/bin/aarch64-linux-gnu-as  /usr/local/bin/aarch64-linux-gnu-as
COPY --from=build /usr/bin/aarch64-linux-gnu-ld  /usr/local/bin/aarch64-linux-gnu-ld
COPY --from=build /usr/bin/aarch64-linux-gnu-gcc /usr/local/bin/aarch64-linux-gnu-gcc
COPY --from=build /usr/bin/qemu-aarch64-static   /usr/local/bin/qemu-aarch64-static

# Copy shared libraries needed by the toolchain binaries
COPY --from=build /usr/lib/aarch64-linux-gnu/    /usr/lib/aarch64-linux-gnu/
COPY --from=build /usr/aarch64-linux-gnu/        /usr/aarch64-linux-gnu/
COPY --from=build /usr/lib/llvm-21/lib/          /usr/lib/llvm-21/lib/

# GCC cross-compiler internal headers (stddef.h, stdarg.h, etc.) and cc1 binary.
# Without these, gcc -S fails on any kernel that uses #include <stddef.h>.
# The probe test in detect_toolchain passes (no headers) but real kernels fail.
COPY --from=build /usr/lib/gcc-cross/            /usr/lib/gcc-cross/
COPY --from=build /usr/lib/gcc/                  /usr/lib/gcc/

RUN useradd -m -u 1000 armgym
WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY arm_gym/ arm_gym/

USER armgym
EXPOSE 7860
CMD ["uvicorn", "arm_gym.env:app", "--host", "0.0.0.0", "--port", "7860"]
