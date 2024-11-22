from setuptools import find_packages, setup

package_name = 'opencv_ewm'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zzy',
    maintainer_email='zzy@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "erweimapub_node = opencv_ewm.erweimapub:main",
            "erweimasub_node = opencv_ewm.erweimasub:main",
            "key_publisher = opencv_ewm.key_publisher:main",
            "depth_receiver = opencv_ewm.shendu:main"
        ],
    },
)
