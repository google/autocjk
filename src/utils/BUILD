load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@python_deps//:requirements.bzl", "requirement")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files([
    "LICENSE",
])

py_library(
    name = "shape",
    srcs = [
        "shape.py",
    ],
    srcs_version = "PY3",
    deps = [
    ],
)

py_library(
    name = "decomposer",
    srcs = [
        "decomposer.py",
    ],
    data = ["//third_party:babelstone_ids"],
    srcs_version = "PY3",
    deps = [
        requirement("PyICU"),
        requirement("absl-py"),
        requirement("networkx"),
        ":region",
        ":shape",
        "//third_party:babelstone",
    ],
)

py_test(
    name = "decomposer_test",
    srcs = [
        "decomposer_test.py",
    ],
    srcs_version = "PY3",
    deps = [
        requirement("decorator"),
        requirement("more_itertools"),
        requirement("networkx"),
        ":decomposer",
        ":ideographic_description_sequence",
        requirement("tensorflow"),
        ":region",
    ],
)

py_library(
    name = "ideographic_description_sequence",
    srcs = [
        "ideographic_description_sequence.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":region",
        requirement("absl-py"),
    ]
)

py_test(
    name = "ideographic_description_sequence_test",
    srcs = [
        "ideographic_description_sequence_test.py",
    ],
    deps = [
        ":ideographic_description_sequence",
        ":region",
        requirement("tensorflow"),
    ],
)

py_library(
    name = "region",
    srcs = [
        "region.py",
    ],
    srcs_version = "PY3",
    deps = [],
)

py_library(
    name = "font_helper",
    srcs = [
        "font_helper.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":region",
        requirement("fonttools"),
    ],
)
