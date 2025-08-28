from conan import ConanFile
from conan.tools.cmake import cmake_layout

class IdentiqueRatioDemo(ConanFile):
    name = "identique_ratio_demo"
    version = "0.1.0"

    settings = "os", "compiler", "build_type", "arch"


    requires = (
        "opencv/4.9.0",
        "zxing-cpp/2.3.0",
        "tensorflow-lite/2.15.0",
        "gtk/system",
    )


    tool_requires = (
        "cmake/3.29.3",
        "ninja/1.11.1",
    )


    generators = ("CMakeDeps", "CMakeToolchain")


    default_options = {
        # --- global ---
        "libpng/*:shared": True,
        "zlib/*:shared": True,
        "libjpeg-turbo/*:shared": True,
        "*:shared": False,

        # --- OpenCV ---
        "opencv/*:contrib": False,
        "opencv/*:nonfree": False,
        "opencv/*:world":   False,
        "opencv/*:with_gtk": True,
        "opencv/*:with_qt": False,
        "opencv/*:with_ffmpeg": False,
        "opencv/*:with_gstreamer": False,
        "opencv/*:with_v4l": True,
        "opencv/*:with_jpeg": "libjpeg-turbo",
        "opencv/*:with_png": True,
        "opencv/*:with_tiff": False,
        "opencv/*:with_openexr": False,
        "opencv/*:with_opencl": False,
        "opencv/*:with_quirc": False,
        "opencv/*:with_eigen": False,
        "opencv/*:with_ipp": False,
        "opencv/*:with_wayland": False,
        # --- ZXing‑cpp ---
        "zxing-cpp/*:build_examples": False,
        "zxing-cpp/*:build_tests":     False,
        "zxing-cpp/*:fuzzers":         False,

        # --- TensorFlow Lite ---
        "tensorflow-lite/*:xnnpack": True,
        "tensorflow-lite/*:ruy":      False,
        "tensorflow-lite/*:eigen":    False,
        "tensorflow-lite/*:fp16":     False,

        "xkbcommon/*:with_wayland": False,
        "xkbcommon/*:with_x11": False,
        "gtk/*:version": 3,
    }

    def layout(self):
        cmake_layout(self)


    def package_info(self):
        self.cpp_info.system_libs = ["dl", "pthread"]
