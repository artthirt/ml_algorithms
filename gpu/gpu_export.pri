INCLUDEPATH += $$PWD

GPU_EXPORTS = "GPU_EXPORTS=\"\""

DEFINES += _USE_GPU $$GPU_EXPORTS

message(gpuexport $$GPUEXPORTS)

HEADERS += $$PWD/convnn2_gpu.h \
    $$PWD/convnn2_mixed.h \
    $$PWD/gpu_mlp.h \
    $$PWD/gpumat.h \
    $$PWD/mlp_mixed.h

LIBS += -L$$DESTDIR/ -lgpu_algorithms
