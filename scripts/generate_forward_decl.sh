#!/bin/bash

echo "#ifndef BVH_FORWARD_DECL_HPP"
echo "#define BVH_FORWARD_DECL_HPP"
echo ""
echo "// Note: This is an automatically generated header."
echo "//       Changes to this file may be lost."
echo ""
echo "namespace bvh {"
echo ""

# Find all template declarations
matches=$(pcregrep --no-filename -Mo "template <.+>\n\b(struct|class)\b \w+ " include/bvh/*.hpp)

# Add semicolons to the end of each declaration
result=$(echo $matches | sed -E 's/\b(class|struct)\b (\w+)/\1 \2;\n/g')

# Cleanup by removing leading spaces from 'template'
result=$(echo "$result" | sed 's/^ template/template/')

# Remove default template arguments
result=$(echo "$result" | sed -E 's/ = (([0-9]+)|\b(false|true))\b//g')

# Print the result to the header
echo "$result"

echo ""
echo "} // namespace bvh"
echo ""
echo "#endif"
