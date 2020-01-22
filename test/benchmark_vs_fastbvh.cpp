#include <cstdio>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include "bvh.hpp"
#include "obj.hpp"

////////////////////////////////////////////////////////////////////////////////////
/// THE FOLLOWING CODE IS AN ADAPTED VERSION OF "RayTracerTest.cpp" FROM Fast-BVH //
////////////////////////////////////////////////////////////////////////////////////

using Vector3 = bvh::Vec3;
using BVH = bvh::Accel<bvh::Triangle>;
using Ray = bvh::Ray;

Vector3 operator ^ (const Vector3& a, const Vector3& b) {
  return cross(a, b);
}

int main(int argc, char** argv) {
  if (argc < 2) {
      std::cout << "Usage: benchmark_vs_fastbvh [--fast-bvh-build] file.obj" << std::endl;
      return 1;
  }

  bool fast = false;
  const char* input_file = NULL;
  for (int i = 1; i < argc; ++i) {
      if (argv[i][0] == '-') {
          if (!strcmp(argv[i], "--fast-bvh-build"))
              fast = true;
          else {
              std::cerr << "Unknown option: '" << argv[i] << "'" << std::endl;
              return 1;
          }
      } else {
          if (input_file) {
              std::cerr << "Scene file specified twice" << std::endl;
              return 1;
          }
          input_file = argv[i];
      }
  }

  if (!input_file) {
      std::cerr << "Missing a command line argument for the scene file" << std::endl;
      return 1;
  }

  // Load mesh from file
  auto objects = obj::load_from_file(input_file);

  // Compute a BVH for this object set
  auto start_tick = std::chrono::high_resolution_clock::now();
  BVH bvh(objects.data(), objects.size());
  bvh.build(fast);
  auto end_tick = std::chrono::high_resolution_clock::now();
  std::cout << "BVH construction took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_tick - start_tick).count() << "ms" << std::endl;

  // Allocate space for some image pixels
  const unsigned int width=8000, height=8000;
  float* pixels = new float[width*height*3];

  // Create a camera from position and focus point
  //Vector3 camera_position(1.6, 1.3, 1.6);
  //Vector3 camera_focus(0,0,0);
  //Vector3 camera_up(0,1,0);

  // For Sponza
  Vector3 camera_position(-1000, 1000, 0);
  Vector3 camera_focus(0,0,0);
  Vector3 camera_up(0,1,0);

  // Camera tangent space
  Vector3 camera_dir = normalize(camera_focus - camera_position);
  Vector3 camera_u = normalize(camera_dir ^ camera_up);
  Vector3 camera_v = normalize(camera_u ^ camera_dir);

  printf("Rendering image (%dx%d)...\n", width, height);
  
  start_tick = std::chrono::high_resolution_clock::now();
  // Raytrace over every pixel
#pragma omp parallel for
  for(size_t i=0; i<width; ++i) {
    for(size_t j=0; j<height; ++j) {
      size_t index = 3*(width * j + i);

      float u = (i+.5f) / (float)(width-1) - .5f;
      float v = (height-1-j+.5f) / (float)(height-1) - .5f;
      float fov = .5f / tanf( 70.f * 3.14159265*.5f / 180.f);

      // This is only valid for square aspect ratio images
      Ray ray(camera_position, normalize(u*camera_u + v*camera_v + fov*camera_dir));

      auto hit = bvh.intersect_closest(ray);
      if(!hit) {
        pixels[index] = pixels[index+1] = pixels[index+2] = 0.f;
      } else {

        // Just for fun, we'll make the color based on the normal
        const Vector3 normal = normalize(objects[hit->first].n);
        const Vector3 color(fabs(normal.x), fabs(normal.y), fabs(normal.z));

        pixels[index  ] = color.x;
        pixels[index+1] = color.y;
        pixels[index+2] = color.z;
      }
    }
  }
  end_tick = std::chrono::high_resolution_clock::now();
  std::cout << "Rendering took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_tick - start_tick).count() << "ms" << std::endl;

  // Output image file (PPM Format)
  printf("Writing out image file: \"render.ppm\"\n");
  FILE *image = fopen("render.ppm", "w");
  fprintf(image, "P6\n%d %d\n255\n", width, height);
  for(size_t j=0; j<height; ++j) {
    for(size_t i=0; i<width; ++i) {
      size_t index = 3*(width * j + i);
      unsigned char r = std::max(std::min(pixels[index  ]*255.f, 255.f), 0.f);
      unsigned char g = std::max(std::min(pixels[index+1]*255.f, 255.f), 0.f);
      unsigned char b = std::max(std::min(pixels[index+2]*255.f, 255.f), 0.f);
      fprintf(image, "%c%c%c", r,g,b);
    }
  }
  fclose(image);

  // Cleanup
  delete[] pixels;
}
