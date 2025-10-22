from setuptools import setup
import os
from glob import glob

package_name = 'race_monitor'

def get_data_files(directory):
    """Get all files in directory, excluding subdirectories"""
    files = []
    for item in glob(os.path.join(directory, '*')):
        if os.path.isfile(item):
            files.append(item)
    return files

setup(
    name=package_name,
    version='2.0.6',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(
            os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'),
         glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'ref_trajectory'),
         get_data_files('ref_trajectory')),
        (os.path.join('share', package_name, 'data'),
         get_data_files('data')),
        (os.path.join('lib', 'python3.10', 'site-packages'),
         ['race_monitor_data.txt']),
    ],
    install_requires=[
        'setuptools',
        'numpy>=1.21.0,<2.0.0',
        'pandas>=1.5.0',
        'scipy>=1.9.0',
        'evo>=1.26.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.12.0',
        'plotly>=5.0.0',
        'psutil>=5.8.0',
        'PyYAML>=6.0',
        'json5>=0.9.0',
        'colorama>=0.4.4',
        'tqdm>=4.64.0',
        'scikit-learn>=1.1.0',
        'statsmodels>=0.13.0'
    ],
    zip_safe=True,
    maintainer='Mohammed Abdeazim',
    maintainer_email='mohammed@azab.io',
    description='ROS2 node for tracking laps, lap times, and race statistics in autonomous racing with EVO trajectory analysis',
    license='MIT',
    extras_require={
        'test': ['pytest>=7.0.0', 'pytest-cov>=4.0.0']
    },
    entry_points={
        'console_scripts': [
            'race_monitor = race_monitor.race_monitor:main',
        ],
    },
)
