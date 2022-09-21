include_directories(.)

add_subdirectory(pybind11) # 已经编译完的pybind11
pybind11_add_module(example example.cpp)
target_link_libraries(
  example

  snark
)
target_include_directories(
  example

  PUBLIC
  ${DEPENDS_DIR}/libsnark
  ${DEPENDS_DIR}/libsnark/depends/libfqfft
)

add_executable(
  main

  main.cpp
)
target_link_libraries(
  main

  snark
)
target_include_directories(
  main

  PUBLIC
  ${DEPENDS_DIR}/libsnark
  ${DEPENDS_DIR}/libsnark/depends/libfqfft
)

pybind11_add_module(test test.cpp)


target_link_libraries(
  test

  snark
  libpython3.8.so
)
target_include_directories(
  test

  PUBLIC
  ${DEPENDS_DIR}/libsnark
  ${DEPENDS_DIR}/libsnark/depends/libfqfft
)

add_executable(
  range

  range.cpp
)
target_link_libraries(
  range

  snark
)
target_include_directories(
  range

  PUBLIC
  ${DEPENDS_DIR}/libsnark
  ${DEPENDS_DIR}/libsnark/depends/libfqfft
)


