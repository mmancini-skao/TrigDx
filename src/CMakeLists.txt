include(FetchContent)
include(FindAVX)
add_library(trigdx reference.cpp lookup.cpp)

if(HAVE_AVX)
    target_compile_definitions(trigdx PUBLIC HAVE_AVX)
    target_compile_options(trigdx PUBLIC -mavx)
endif()

if(HAVE_AVX2)
    target_compile_definitions(trigdx PUBLIC HAVE_AVX2)
    target_compile_options(trigdx PUBLIC -mavx2)
endif()

target_include_directories(trigdx PUBLIC ${PROJECT_SOURCE_DIR}/include)

if(HAVE_AVX)
  target_sources(trigdx PRIVATE lookup_avx.cpp)
endif()

if(TRIGDX_USE_MKL)
  find_package(MKL REQUIRED)
  target_sources(trigdx PRIVATE mkl.cpp)
  target_link_libraries(trigdx PRIVATE MKL::MKL)
endif()

if(TRIGDX_USE_GPU)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(gpu SHARED gpu/gpu.cu)
  target_sources(trigdx PRIVATE gpu.cpp)
  target_link_libraries(trigdx PRIVATE CUDA::cudart)
  target_link_libraries(trigdx PRIVATE gpu)
endif()

if(TRIGDX_USE_XSIMD)
  # Requires XSIMD > 13 for architecture independent dispatching
  find_package(xsimd 13 QUIET)
  if(NOT TARGET xsimd)
    FetchContent_Declare(
      xsimd
      GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
      GIT_TAG 13.2.0)
    FetchContent_MakeAvailable(xsimd)
  endif()
  target_sources(trigdx PRIVATE lookup_xsimd.cpp)
  target_link_libraries(trigdx PRIVATE xsimd)
endif()

target_include_directories(
  trigdx INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                   $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)
