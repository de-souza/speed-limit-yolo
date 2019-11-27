TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        src/main.c

unix: LIBS += -L$$PWD/../darknet/ -ldarknet -ltesseract -llept

INCLUDEPATH += $$PWD/../darknet/include
DEPENDPATH += $$PWD/../darknet/include

unix: PRE_TARGETDEPS += $$PWD/../darknet/libdarknet.a

QMAKE_POST_LINK = \
        $$QMAKE_COPY_DIR $$shell_path($$PWD/cfg) $$shell_path($$OUT_PWD) $$escape_expand(\n\t) \
        $$QMAKE_COPY_DIR $$shell_path($$PWD/data) $$shell_path($$OUT_PWD)
