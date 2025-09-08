FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set the working directory
WORKDIR /EDGS

# Install all system dependencies including Node.js
RUN apt-get update && \
  apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    ca-certificates \
    gnupg && \
  mkdir -p /etc/apt/keyrings && \
  curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
  echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_18.x nodistro main" > /etc/apt/sources.list.d/nodesource.list && \
  apt-get update && \
  apt-get install -y nodejs && \
  rm -rf /var/lib/apt/lists/*

# Copy only essential files for cloning submodules first (e.g., .gitmodules)
# Or, if submodules are public, you might not need to copy anything specific for this step
# For simplicity, we'll copy everything, but this could be optimized
COPY . .

# Initialize and update submodules
RUN git submodule init && git submodule update --recursive

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
  bash /tmp/miniconda.sh -b -p /opt/conda && \
  rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Create the conda environment and install dependencies
# Accept Anaconda TOS before using conda
RUN conda init bash && \
  conda config --set always_yes yes --set changeps1 no && \
  conda config --add channels defaults && \
  conda config --set channel_priority strict && \
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
# Now you can safely create your environment
RUN conda create -y -n edgs python=3.10 pip && \
  conda clean -afy && \
  echo "source activate edgs" > ~/.bashrc

# Set CUDA architectures to compile for
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX"

# Activate the environment and install Python dependencies
RUN /bin/bash -c "source activate edgs && \
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
  pip install -e ./submodules/gaussian-splatting/submodules/diff-gaussian-rasterization && \
  pip install -e ./submodules/gaussian-splatting/submodules/simple-knn && \
  pip install pycolmap wandb hydra-core tqdm torchmetrics lpips matplotlib rich plyfile imageio imageio-ffmpeg opencv-python && \
  pip install -e ./submodules/RoMa && \
  pip install gradio plotly scikit-learn moviepy==2.1.1 ffmpeg open3d jupyterlab matplotlib"

# Install Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Create a non-root user
RUN useradd -m -s /bin/bash claude_user && \
    chown -R claude_user:claude_user /EDGS && \
    chown -R claude_user:claude_user /opt/conda

# Switch to non-root user
USER claude_user

# Set up conda for the new user and make it the default
RUN echo "source activate edgs" >> ~/.bashrc

# Set environment variables to make conda environment active by default
ENV CONDA_DEFAULT_ENV=edgs
ENV CONDA_PREFIX=/opt/conda/envs/edgs
ENV PATH=/opt/conda/envs/edgs/bin:$PATH
ENV CONDA_PYTHON_EXE=/opt/conda/envs/edgs/bin/python

# Create an entrypoint script that ensures conda env is active
RUN echo '#!/bin/bash\nsource /opt/conda/etc/profile.d/conda.sh\nconda activate edgs\nexec "$@"' > /home/claude_user/entrypoint.sh && \
    chmod +x /home/claude_user/entrypoint.sh

# Expose the port for Gradio
EXPOSE 7862

# Set the entrypoint to ensure conda env is always active
ENTRYPOINT ["/home/claude_user/entrypoint.sh"]

# Default command
CMD ["tail", "-f", "/dev/null"]