from setuptools import find_packages, setup

package_name = 'opencv_color'

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
            "red_detectionpub_node = opencv_color.red_detectionpub:main",
            "red_detectionsub_node = opencv_color.red_detectionsub:main",
            "point_publisher = opencv_color.mubiao:main",
            "lower_body_detector = opencv_color.xiabanspub:main",
        ],
    },
)
