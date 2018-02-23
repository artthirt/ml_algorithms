TARGET = gpu_algorithms
TEMPLATE = lib

CONFIG += c++11
CONFIG -= qt

DESTDIR = ../../

unix*{
    QMAKE_CC = gcc-6
    QMAKE_CXX = g++-6

    QMAKE_CFLAGS += -fPIC -fopenmp
    QMAKE_CXXFLAGS += -fPIC -fopenmp

    LIBS += -lgomp
    INCLUDEPATH += /usr/include/c++/6/
}

win32{
    GPU_EXPORTS = "GPU_EXPORTS=__declspec(dllexport)"

    QMAKE_CFLAGS += /openmp
    QMAKE_CXXFLAGS += /openmp

}else{
    GPU_EXPORTS = "GPU_EXPORTS=\"__attribute__ ((visibility (\\\"default\\\")))\""
}

CONFIG(debug, debug|release){
    OBJECTS_DIR = tmp/debug/obj
}else{
    OBJECTS_DIR = tmp/release/obj
}

include(../ct/ct.pri)
include(gpu.pri)
