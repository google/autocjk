load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@pip_deps//:requirements.bzl", "requirement")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files([
    "LICENSE",
])

py_library(
    name = "imageutils",
    srcs = [
        "imageutils.py",
    ],
    srcs_version = "PY3",
    deps = [
        requirement("image"),
        requirement("imagehash"),
        requirement("numpy"),
    ],
)

py_binary(
    name = "evaluate_font",
    srcs = [
        "evaluate_font.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":imageutils",
        "//models:generator_lib",
        "//src/utils:decomposer",
        "//src/utils:font_helper",
        "//src/utils:ideographic_description_sequence",
        requirement("imagehash"),
        requirement("image"),
    ],
)

py_binary(
    name = "evaluate_glyphwiki",
    srcs = [
        "evaluate_glyphwiki.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":imageutils",
        "//models:generator_lib",
        "//src/utils:decomposer",
        "//src/utils:font_helper",
        "//src/utils:ideographic_description_sequence",
        requirement("imagehash"),
        requirement("image"),
    ],
)

py_library(
    name = "model",
    srcs = [
        "model.py",
    ],
    srcs_version = "PY3",
    deps = [
        requirement("backcall"),
        requirement("ipython"),
        requirement("ipython_genutils"),
        requirement("pickleshare"),
        requirement("prompt_toolkit"),
        requirement("traitlets"),
        requirement("wcwidth"),
    ],
)

py_binary(
    name = "make_dataset",
    srcs = [
        "make_dataset.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":imageutils",
        "//src/utils:decomposer",
        "//src/utils:font_helper",
        "//src/utils:ideographic_description_sequence",
        "//src/utils:region",
        requirement("absl-py"),
        requirement("imagehash"),
        requirement("image"),
    ],
)

py_test(
    name = "make_dataset_test",
    srcs = ["make_dataset_test.py"],
    data = [
        "//src/testutils:fonts",
        "//src/testutils:expected_training_data",
        # Include the binary as data.
        ":make_dataset",
    ],
    srcs_version = "PY3",
    deps = [
        requirement("imagehash"),
    ],
)

py_binary(
    name = "main",
    srcs = [
        "main.py",
    ],
    deps = [
        ":imageutils",
        "//models:generator_lib",
        requirement("absl-py"),
    ],
)

py_test(
    name = "main_test",
    srcs = ["main_test.py"],
    data = [
        "//src/testutils:fonts",
        "//src/testutils:expected_predictions",
        # Include the binary as data.
        ":main",
    ],
    srcs_version = "PY3",
    deps = [
        requirement("imagehash"),
    ],
)
