load("@rules_python//python:defs.bzl", "py_library")

package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "babelstone_ids",
    srcs = [
        "IDS.TXT",
    ],
)

py_library(
    name = "babelstone",
    srcs = [
        "babelstone.py",
    ],
    data = [":babelstone_ids"],
    srcs_version = "PY3",
)
