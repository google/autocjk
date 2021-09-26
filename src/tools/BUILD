load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@python_deps//:requirements.bzl", "requirement")

py_binary(
    name="make_dataset",
    srcs=[
        "make_dataset.py",
    ],
    srcs_version="PY3",
    deps=[
        ":make_dataset_lib",
        requirement("absl-py"),
    ],
)

py_test(
    name="make_dataset_test",
    srcs=["make_dataset_test.py"],
    srcs_version="PY3",
    data=[
        "//src/testutils:fonts", 
        ":make_dataset"
    ],
    deps=[
        requirement("tensorflow"),
    ],
)

py_library(
    name="make_dataset_lib",
    srcs=[
        "make_dataset_lib.py",
    ],
    srcs_version="PY3",
    deps=[
        "//src:imageutils",
        "//src/utils:decomposer",
        "//src/utils:font_helper",
        "//src/utils:ideographic_description_sequence",
        "//src/utils:region",
        "//src/utils:shape",
        requirement("absl-py"),
        requirement("fontTools"),
        requirement("imagehash"),
        requirement("networkx"),
    ],
)

py_test(name="make_dataset_lib_test",
        srcs=["make_dataset_lib_test.py"],
        srcs_version="PY3",
        data=[
            "//src/testutils:fonts",
            "//src/testutils/corpus:split-1",
            "//src/testutils/corpus:split-2",
        ],
        deps=[
            ":make_dataset_lib",
            requirement("Pillow"),
            requirement("imagehash"),
            requirement("tensorflow"),
            requirement("absl-py"),
        ])
