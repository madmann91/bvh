# Finds the "restrict" keyword that is accepted by the compiler, if any.
# The arguments are:
#
#   RESTRICT_KEYWORD : The "restrict" keyword that can be used with the compiler.
#                      If no such keyword exists, returns the empty string. 
function (find_restrict_keyword restrict_keyword)
    foreach (keyword "restrict" "__restrict" "__restrict__")
        try_compile(status ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/cmake/test_restrict.cpp COMPILE_DEFINITIONS -Drestrict=${keyword} CXX_EXTENSIONS ON)
        if (status)
            set(${restrict_keyword} ${keyword} PARENT_SCOPE)
            return()
        endif ()
    endforeach ()
    set(${restrict_keyword} "" PARENT_SCOPE)
endfunction ()
