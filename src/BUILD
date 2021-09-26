load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@python_deps//:requirements.bzl", "requirement")

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
        requirement("absl-py"),
        requirement("Pillow"),
        requirement("imagehash"),
        requirement("numpy"),
        requirement("tensorflow"),
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

py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    srcs_version = "PY3",
    data = [
        "//src/testutils/corpus:split-1",
        "//src/testutils/corpus:split-3",
        "//src/testutils:fonts",
    ],
    deps = [
        ":model",
        "imageutils",
        requirement("tensorflow"),
        requirement("Pillow"),
        requirement("matplotlib"),
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
        "//src/utils:decomposer",
        "//src/utils:ideographic_description_sequence",
        "//src/utils:region",
        requirement("absl-py"),
        requirement("Pillow"),
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
        requirement("Pillow"),
        requirement("tensorflow"),
        requirement("imagehash"),
    ],
)

py_test(
    name = "pillow_test",
    srcs = ["pillow_test.py"],
    srcs_version = "PY3",
    deps = [
        requirement("tensorflow"),
        requirement("Pillow"),
    ],
)