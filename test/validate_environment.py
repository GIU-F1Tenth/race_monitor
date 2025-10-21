#!/usr/bin/env python3

"""
Race Monitor Environment Validation Script

This script validates that all necessary dependencies are installed
and properly configured for the race_monitor package.

Usage:
    python3 validate_environment.py

"""

import sys
import subprocess
import importlib
import os
from pathlib import Path


def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(
            f"✅ Python {version.major}.{version.minor}.{version.micro} is supported")
        return True
    else:
        print(
            f"❌ Python {version.major}.{version.minor}.{version.micro} is not supported. Requires Python 3.10+")
        return False


def check_ros2_installation():
    """Check if ROS2 is installed and sourced."""
    print("\n🤖 Checking ROS2 installation...")

    # Check if ROS_DISTRO is set
    ros_distro = os.environ.get('ROS_DISTRO')
    if ros_distro:
        print(f"✅ ROS2 {ros_distro} environment detected")
        return True
    else:
        print("❌ ROS2 environment not sourced. Please run: source /opt/ros/<distro>/setup.bash")
        return False


def check_python_dependency(package_name, import_name=None):
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = package_name

    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name} - Not installed")
        return False


def check_python_dependencies():
    """Check all required Python dependencies."""
    print("\n📦 Checking Python dependencies...")

    core_dependencies = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scipy', 'scipy'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('plotly', 'plotly'),
        ('psutil', 'psutil'),
        ('PyYAML', 'yaml'),
        ('colorama', 'colorama'),
        ('tqdm', 'tqdm'),
        ('scikit-learn', 'sklearn'),
        ('statsmodels', 'statsmodels'),
    ]

    # ROS2-specific dependencies (optional in pure Python environments)
    ros2_dependencies = [
        ('tf_transformations', 'tf_transformations'),
    ]

    all_good = True

    # Check core dependencies (required)
    print("  Core dependencies:")
    for package_name, import_name in core_dependencies:
        if not check_python_dependency(package_name, import_name):
            all_good = False

    # Check ROS2 dependencies (optional)
    print("  ROS2 dependencies (optional in pure Python environments):")
    ros2_available = True
    for package_name, import_name in ros2_dependencies:
        if not check_python_dependency(package_name, import_name):
            ros2_available = False
            print(
                f"    ℹ️  {package_name} not available (normal in pure Python environments)")

    if ros2_available:
        print("  ✅ ROS2 dependencies are available")
    else:
        print(
            "  ⚠️  ROS2 dependencies not available (install via apt in ROS2 environments)")

    return all_good


def check_evo_submodule():
    """Check if EVO submodule is properly initialized."""
    print("\n📊 Checking EVO submodule...")

    evo_path = Path(__file__).parent / "evo"
    if evo_path.exists() and any(evo_path.iterdir()):
        print("✅ EVO submodule directory exists and has content")

        # Try to import EVO
        try:
            sys.path.insert(0, str(evo_path))
            import evo.core.trajectory
            print("✅ EVO library can be imported")
            return True
        except ImportError as e:
            print(f"❌ EVO library import failed: {e}")
            return False
    else:
        print(
            "❌ EVO submodule not initialized. Run: git submodule update --init --recursive")
        return False


def check_ros2_packages():
    """Check if required ROS2 packages are available."""
    print("\n🔧 Checking ROS2 packages...")

    ros2_packages = [
        'rclpy',
        'std_msgs',
        'nav_msgs',
        'geometry_msgs',
        'visualization_msgs',
        'ackermann_msgs',
        'tf2_ros'
    ]

    all_good = True
    for package in ros2_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(
                f"❌ {package} - Install with: sudo apt install ros-$ROS_DISTRO-{package.replace('_', '-')}")
            all_good = False

    return all_good


def check_colcon():
    """Check if colcon build tools are available."""
    print("\n🔨 Checking build tools...")

    try:
        result = subprocess.run(['colcon', 'version-check'],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ colcon build tools available")
            return True
        else:
            print("❌ colcon not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ colcon not found. Install with: sudo apt install python3-colcon-common-extensions")
        return False


def check_workspace_structure():
    """Check if workspace has proper structure."""
    print("\n📁 Checking workspace structure...")

    base_path = Path(__file__).parent
    required_dirs = [
        'race_monitor',
        'config',
        'launch',
        'evo'
    ]

    required_files = [
        'setup.py',
        'package.xml',
        'requirements.txt'
    ]

    all_good = True

    for directory in required_dirs:
        dir_path = base_path / directory
        if dir_path.exists():
            print(f"✅ {directory}/ directory")
        else:
            print(f"❌ {directory}/ directory missing")
            all_good = False

    for file_name in required_files:
        file_path = base_path / file_name
        if file_path.exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} missing")
            all_good = False

    return all_good


def print_installation_help():
    """Print helpful installation commands."""
    print("\n" + "="*60)
    print("🚀 INSTALLATION HELP")
    print("="*60)
    print()
    print("If you have missing dependencies, try these commands:")
    print()
    print("# Install Python dependencies:")
    print("pip3 install -r requirements.txt")
    print()
    print("# Install ROS2 dependencies:")
    print("sudo apt update")
    print("sudo apt install \\")
    print("  ros-$ROS_DISTRO-rclpy \\")
    print("  ros-$ROS_DISTRO-std-msgs \\")
    print("  ros-$ROS_DISTRO-nav-msgs \\")
    print("  ros-$ROS_DISTRO-geometry-msgs \\")
    print("  ros-$ROS_DISTRO-visualization-msgs \\")
    print("  ros-$ROS_DISTRO-ackermann-msgs \\")
    print("  ros-$ROS_DISTRO-tf2-ros \\")
    print("  ros-$ROS_DISTRO-tf-transformations \\")
    print("  python3-colcon-common-extensions")
    print()
    print("# Initialize EVO submodule:")
    print("git submodule update --init --recursive")
    print()
    print("# Build the package:")
    print("colcon build --packages-select race_monitor")
    print("source install/setup.bash")


def main():
    """Run all validation checks."""
    print("Race Monitor Environment Validation")
    print("="*50)

    checks = [
        check_python_version,
        check_ros2_installation,
        check_python_dependencies,
        check_evo_submodule,
        check_ros2_packages,
        check_colcon,
        check_workspace_structure
    ]

    results = []
    for check in checks:
        results.append(check())

    print("\n" + "="*50)
    print("📋 VALIDATION SUMMARY")
    print("="*50)

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✅ All checks passed ({passed}/{total})")
        print("\n🎉 Your environment is ready for race_monitor development!")
    else:
        print(f"❌ {total - passed} checks failed ({passed}/{total} passed)")
        print("\n❗ Please fix the issues above before proceeding.")
        print_installation_help()

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
