INCLUDEPATH += $$PWD

GPU_EXPORTS = "GPU_EXPORTS=\"\""

DEFINES += _USE_GPU $$GPU_EXPORTS

message(gpuexport $$GPUEXPORTS)

HEADERS += gpumat.h \
        gpu_mlp.h \
        convnn2_gpu.h

LIBS += -L$$DESTDIR/ -lgpu_algorithms
