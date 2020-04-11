include(CheckCXXSourceCompiles)

# Finds the "restrict" keyword that is accepted by the compiler, if any.
# `restrict_keyword` is a variable that is set to the "restrict" keyword
# that can is supported by the compiler in use. If no such keyword exists,
# it is set to the empty string. 
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
