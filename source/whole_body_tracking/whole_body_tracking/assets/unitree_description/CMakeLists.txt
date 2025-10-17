cmake_minimum_required(VERSION 3.10)
project(unitree_description)

find_package(ament_cmake REQUIRED)

install(
        DIRECTORY meshes urdf mjcf launch
        DESTINATION share/${PROJECT_NAME}
)

ament_package()
