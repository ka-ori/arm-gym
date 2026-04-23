# Cut 1 fix: LLVM 21 Olympus scheduler for Neoverse V3 -- not clang-18.
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG LLVM_VERSION=21

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg lsb-release software-properties-common \
      build-essential python3 python3-pip python3-venv \
      qemu-user-static binutils-aarch64-linux-gnu gcc-aarch64-linux-gnu \
      time \
  && rm -rf /var/lib/apt/lists/*

# Kitware / LLVM apt repo for llvm-21 (Olympus scheduler).
# Fallback logic in compile_baseline.py disclosures V2 proxy if V3 unavailable.
RUN curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
  && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/noble/ llvm-toolchain-noble-${LLVM_VERSION} main" \
      > /etc/apt/sources.list.d/llvm.list \
  && apt-get update \
  && (apt-get install -y --no-install-recommends \
        clang-${LLVM_VERSION} llvm-${LLVM_VERSION} llvm-${LLVM_VERSION}-tools lld-${LLVM_VERSION} \
      || (echo "LLVM ${LLVM_VERSION} unavailable, falling back to latest" \
          && apt-get install -y --no-install-recommends clang llvm llvm-tools lld)) \
  && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/lib/llvm-${LLVM_VERSION}/bin:${PATH}
WORKDIR /work
COPY pyproject.toml /work/
RUN pip3 install --break-system-packages -e . || pip3 install --break-system-packages .
COPY . /work

CMD ["python3", "-m", "uvicorn", "arm_gym.env:app", "--host", "0.0.0.0", "--port", "8000"]
