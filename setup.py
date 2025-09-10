from setuptools import setup
import os
from glob import glob

package_name = 'race_monitor'

setup(
    name=package_name,
    version='2.0.5',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools', 'evo', 'psutil', 'ackermann_msgs'],
    zip_safe=True,
    maintainer='Mohammed Azab',
    maintainer_email='mohammed@azab.io',
    description='ROS2 node for tracking laps, lap times, and race statistics in autonomous racing with EVO trajectory analysis',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'race_monitor = race_monitor.race_monitor:main',
        ],
    },
)
