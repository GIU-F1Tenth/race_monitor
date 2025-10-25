# Race Monitor CI Docker Image
# Ubuntu 22.04 + ROS2 Humble + Python dependencies pre-installed
# This image is used in CI to speed up builds by avoiding repeated dependency installation

FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

# Set up locale
RUN apt-get update && apt-get install -y \
    locales \
    && locale-gen en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Install basic tools and ROS2 repository
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    gnupg2 \
    lsb-release \
    git \
    && rm -rf /var/lib/apt/lists/*

# Add ROS2 apt repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list

# Install ROS2 Humble and core packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    build-essential \
    python3-rosdep \
    ros-humble-rclpy \
    ros-humble-std-msgs \
    ros-humble-nav-msgs \
    ros-humble-geometry-msgs \
    ros-humble-visualization-msgs \
    ros-humble-ackermann-msgs \
    ros-humble-tf2-ros \
    ros-humble-ros2cli \
    ros-humble-ros2launch \
    ros-humble-ros2run \
    && rm -rf /var/lib/apt/lists/*

# Remove conflicting system transforms3d package
RUN apt-get update && \
    (apt-get purge -y python3-transforms3d 2>/dev/null || true) && \
    rm -rf /usr/lib/python3/dist-packages/transforms3d* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get download ros-humble-tf-transformations && \
    dpkg --force-depends -i ros-humble-tf-transformations*.deb && \
    rm ros-humble-tf-transformations*.deb && \
    rm -rf /var/lib/apt/lists/*

# Initialize rosdep (as root in Docker)
RUN rosdep init || echo "rosdep already initialized" \
    && rosdep update

# Add ROS2 binaries to PATH
ENV PATH="/opt/ros/humble/bin:${PATH}"
ENV PYTHONPATH="/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages"
ENV LD_LIBRARY_PATH="/opt/ros/humble/lib"
ENV AMENT_PREFIX_PATH="/opt/ros/humble"
ENV CMAKE_PREFIX_PATH="/opt/ros/humble"

# Verify ros2 command is available
RUN which ros2 && ros2 --help | head -5 || (echo "✗ ros2 command not found" && exit 1)

# Copy Python requirements and constraints
COPY requirements.txt constraints.txt /tmp/

# Install Python dependencies with constraints
RUN python3 -m pip install --upgrade pip && \
    pip3 install -c /tmp/constraints.txt -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt /tmp/constraints.txt

# Copy and install evo from submodule
COPY evo/ /tmp/evo/
RUN cd /tmp/evo && \
    pip3 install . && \
    cd / && \
    rm -rf /tmp/evo

# Verify evo installation
RUN python3 -c "import evo; print(f'✓ evo {evo.__version__} installed successfully')" || \
    echo "⚠️ evo installation verification failed"

# Verify tf_transformations is available
RUN echo "Verifying tf_transformations installation..." && \
    bash -c "source /opt/ros/humble/setup.bash && python3 -c 'import tf_transformations; print(\"✓ tf_transformations available\")'" || \
    echo "⚠ tf_transformations verification failed"

# Source ROS2 setup in bashrc for interactive sessions
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]