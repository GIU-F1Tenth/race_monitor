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
    ros-humble-desktop \
    python3-rosdep \
    python3-colcon-common-extensions \
    build-essential \
    ros-humble-rclpy \
    ros-humble-std-msgs \
    ros-humble-nav-msgs \
    ros-humble-geometry-msgs \
    ros-humble-visualization-msgs \
    ros-humble-ackermann-msgs \
    ros-humble-tf2-ros \
    ros-humble-tf-transformations \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get purge -y python3-transforms3d || true \
    && rm -rf /usr/lib/python3/dist-packages/transforms3d*

# Initialize rosdep
RUN rosdep init && rosdep update

# Copy Python requirements and constraints
COPY requirements.txt constraints.txt /tmp/

# Install Python dependencies with constraints
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt /tmp/constraints.txt

# Verify critical dependencies are available
RUN python3 -c "import transforms3d; import numpy; print(f'✓ transforms3d {transforms3d.__version__} with NumPy {numpy.__version__}')"

# Source ROS2 setup in bashrc for interactive sessions
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
