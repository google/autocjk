package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files([
    "LICENSE",
])

filegroup(
    name = "generator",
    srcs = [
        "generator.h5",
    ],
)

py_library(
    name = "generator_lib",
    srcs = [
        "generator_lib.py",
    ],
    data = [":generator"],
    srcs_version = "PY3",
)
