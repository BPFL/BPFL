include_directories(.)

add_subdirectory(pybind11)
pybind11_add_module(main main.cpp)

target_link_libraries(
  main

  snark
  libpython3.8.so
)
target_include_directories(
  main

  PUBLIC
  ${DEPENDS_DIR}/libsnark
  ${DEPENDS_DIR}/libsnark/depends/libfqfft
)



