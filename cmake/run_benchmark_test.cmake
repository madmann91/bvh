execute_process(COMMAND ${benchmark_EXECUTABLE} ${benchmark_OPTIONS} RESULT_VARIABLE status)
if (NOT status STREQUAL "0")
    message(FATAL_ERROR "Error while running the benchmarking tool.")
endif ()
if (NOT ImageMagick_compare_EXECUTABLE STREQUAL "")
    execute_process(
        COMMAND
            ${ImageMagick_compare_EXECUTABLE}
            -metric MSE -compose Src -highlight-color White -lowlight-color Black
            ${benchmark_OUTPUT}
            ${benchmark_REFERENCE}
            ${benchmark_DIFFERENCE_RESULT}
        RESULT_VARIABLE status)
    if (NOT status STREQUAL "0")
        message(FATAL_ERROR "Comparison between '${benchmark_OUTPUT}' and '${benchmark_REFERENCE}' failed.")
    else ()
        message(STATUS "The two images '${benchmark_OUTPUT}' and '${benchmark_REFERENCE}' are identical.")
    endif ()
endif ()
