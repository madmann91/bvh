include(CheckCXXSourceCompiles)

# Finds the "restrict" keyword that is accepted by the compiler, if any.
# The arguments are:
#
#   RESTRICT_KEYWORD : The "restrict" keyword that can be used with the compiler.
#                      If no such keyword exists, returns the empty string. 
function (find_restrict_keyword restrict_keyword)
    foreach (keyword "restrict" "__restrict" "__restrict__")
        check_cxx_source_compiles("void f(int* ${keyword} p) { (void)p; } int main() { return 0; }" HAS_RESTRICT_KEYWORD_${keyword})
        if (HAS_RESTRICT_KEYWORD_${keyword})
            set(${restrict_keyword} ${keyword} PARENT_SCOPE)
            return()
        endif ()
    endforeach ()
    set(${restrict_keyword} "" PARENT_SCOPE)
endfunction ()
