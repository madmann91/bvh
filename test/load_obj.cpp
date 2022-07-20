#include "load_obj.h"

#include <fstream>
#include <cassert>
#include <cstdlib>

using namespace bvh::v2;

namespace obj {

inline void remove_eol(char* ptr) {
    int i = 0;
    while (ptr[i]) i++;
    i--;
    while (i > 0 && std::isspace(ptr[i])) {
        ptr[i] = '\0';
        i--;
    }
}

inline char* strip_spaces(char* ptr) {
    while (std::isspace(*ptr)) ptr++;
    return ptr;
}

inline std::optional<int> read_index(char** ptr) {
    char* base = *ptr;

    // Detect end of line (negative indices are supported) 
    base = strip_spaces(base);
    if (!std::isdigit(*base) && *base != '-')
        return std::nullopt;

    int index = std::strtol(base, &base, 10);
    base = strip_spaces(base);

    if (*base == '/') {
        base++;

        // Handle the case when there is no texture coordinate
        if (*base != '/')
            std::strtol(base, &base, 10);

        base = strip_spaces(base);

        if (*base == '/') {
            base++;
            std::strtol(base, &base, 10);
        }
    }

    *ptr = base;
    return std::make_optional(index);
}

template <typename T>
inline std::vector<Tri<T, 3>> load_from_stream(std::istream& is) {
    static constexpr size_t max_line = 1024;
    char line[max_line];

    std::vector<Vec<T, 3>> vertices;
    std::vector<Tri<T, 3>> triangles;

    while (is.getline(line, max_line)) {
        char* ptr = strip_spaces(line);
        if (*ptr == '\0' || *ptr == '#')
            continue;
        remove_eol(ptr);
        if (*ptr == 'v' && std::isspace(ptr[1])) {
            auto x = std::strtof(ptr + 1, &ptr);
            auto y = std::strtof(ptr, &ptr);
            auto z = std::strtof(ptr, &ptr);
            vertices.emplace_back(x, y, z);
        } else if (*ptr == 'f' && std::isspace(ptr[1])) {
            Vec<T, 3> points[2];
            ptr += 2;
            for (size_t i = 0; ; ++i) {
                if (auto index = read_index(&ptr)) {
                    size_t j = *index < 0 ? vertices.size() + *index : *index - 1;
                    assert(j < vertices.size());
                    auto v = vertices[j];
                    if (i >= 2) {
                        triangles.emplace_back(points[0], points[1], v);
                        points[1] = v;
                    } else {
                        points[i] = v;
                    }
                } else {
                    break;
                }
            }
        }
    }

    return triangles;
}

template <typename T>
inline std::vector<Tri<T, 3>> load_from_file(const std::string& file) {
    std::ifstream is(file);
    if (is)
        return load_from_stream<T>(is);
    return std::vector<Tri<T, 3>>();
}

} // namespace obj

template <typename T>
std::vector<Tri<T, 3>> load_obj(const std::string& file) {
    return obj::load_from_file<T>(file);
}

template std::vector<Tri<float, 3>> load_obj(const std::string&);
template std::vector<Tri<double, 3>> load_obj(const std::string&);
