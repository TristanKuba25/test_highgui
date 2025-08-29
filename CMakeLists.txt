cmake_minimum_required(VERSION 3.18)
project(identique_ratio_demo LANGUAGES CXX)

set(CMAKE_INSTALL_PREFIX "/usr" CACHE PATH "Install prefix" FORCE)

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio imgcodecs highgui)
find_package(ZXing  REQUIRED CONFIG)
find_package(tensorflowlite REQUIRED CONFIG)

find_package(Threads REQUIRED)          # pthread
#find_library(DL_LIB dl REQUIRED)        # dlopen/dlsym
add_executable(identique_ratio src/identique_ratio.cpp)

target_link_libraries(identique_ratio PRIVATE
    opencv_core
    opencv_imgproc
    opencv_videoio
    opencv_imgcodecs
    opencv_highgui
    ZXing::ZXing
    tensorflow::tensorflowlite
    Threads::Threads
    ${CMAKE_DL_LIBS}
)
target_link_options(identique_ratio PRIVATE -Wl,--exclude-libs,ALL)
include(InstallRequiredSystemLibraries)
set(CPACK_GENERATOR DEB)
set(CPACK_PACKAGE_NAME    "identique-ratio")
set(CPACK_PACKAGE_VERSION "0.1.0")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Tsol")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.29), libstdc++6 (>= 9)")

install(TARGETS identique_ratio RUNTIME DESTINATION bin)

#install(FILES ${CMAKE_SOURCE_DIR}/models/detection-precision-npu-2025-08-05T06-13-14.743Z_channel_ptq_vvip.tflite DESTINATION bin/models)
install(FILES ${CMAKE_SOURCE_DIR}/headers/anchors_v3.h DESTINATION bin/headers)
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")
include(CPack)
