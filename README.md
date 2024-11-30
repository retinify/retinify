<p align="center">
<img img width="100%" src="./docs/logo.png" alt="retinify">
</p>

# retinify: Open Source AI Stereo Vision
retinify is an advanced AI-powered stereo vision library designed for robotics, offering seamless integration and precise 3D perception.

## Features
![video]()
- **High-Precision**
- **Real-Time**
- **Multiple Cameras (monocular and stereo)**
- **Lightweight GUI**
- **ROS2 Support**

## Install
To fulfill the dependencies, please run the following setup script:
```
sudo ./setup.sh
```
After running the script, please proceed with the following build process:
```
INSTALL_DIR=/path/to/install
```
```
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR ..
make install
```
```
echo 'export CMAKE_PREFIX_PATH=$INSTALL_DIR:$CMAKE_PREFIX_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$INSTALL_DIR/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## License
This project is provided under a dual-license model:

- **AGPL License**: This OSI-approved open-source license is perfect for students, enthusiasts, and non-commercial users. It allows free use, modification, and distribution of the software, as long as any changes are shared under the same license.

- **Enterprise License**: For commercial use, the Enterprise License removes the restrictions of AGPL-3.0, allowing businesses to integrate the software and AI models into commercial products and services.

## Contact