cmake_minimum_required(VERSION 3.16)
project(jev-photos CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(TP ${CMAKE_SOURCE_DIR}/third_party)

# Headers fetched by scripts/bootstrap-headers.sh when the -dev packages are not installed.
if(EXISTS ${TP}/sysroot/usr/include/X11/Xlib.h)
  include_directories(SYSTEM ${TP}/sysroot/usr/include)
  list(APPEND CMAKE_PREFIX_PATH ${TP}/sysroot/usr)
  set(CMAKE_REQUIRED_INCLUDES ${TP}/sysroot/usr/include)
endif()

# GLFW (X11 backend; it dlopens libX11/libGL at runtime).
set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_X11 ON CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${TP}/glfw EXCLUDE_FROM_ALL)
if(EXISTS ${TP}/sysroot/usr/include/X11/Xlib.h)
  target_include_directories(glfw PRIVATE ${TP}/sysroot/usr/include)
endif()

set(IMGUI ${TP}/imgui)
add_library(imgui STATIC
  ${IMGUI}/imgui.cpp ${IMGUI}/imgui_draw.cpp ${IMGUI}/imgui_tables.cpp ${IMGUI}/imgui_widgets.cpp
  ${IMGUI}/backends/imgui_impl_glfw.cpp ${IMGUI}/backends/imgui_impl_opengl3.cpp)
target_include_directories(imgui PUBLIC ${IMGUI})
target_link_libraries(imgui PUBLIC glfw)

# SQLite amalgamation with FTS5 (trigram tokenizer for CJK-friendly search).
add_library(sqlite3 STATIC ${TP}/sqlite/sqlite3.c)
target_include_directories(sqlite3 PUBLIC ${TP}/sqlite)
target_compile_definitions(sqlite3 PRIVATE SQLITE_ENABLE_FTS5 SQLITE_THREADSAFE=1 SQLITE_DQS=0 SQLITE_OMIT_LOAD_EXTENSION)
set_target_properties(sqlite3 PROPERTIES C_VISIBILITY_PRESET hidden)

# Everything except the UI, shared by the app and the tests.
add_library(jevcore STATIC
  src/util.cpp src/config.cpp src/http.cpp src/meta.cpp src/genmeta.cpp src/dates.cpp src/location.cpp
  src/jev.cpp src/vision.cpp src/thumbs.cpp src/clip.cpp src/llm.cpp src/search.cpp src/db.cpp src/dupes.cpp src/pipeline.cpp)
target_include_directories(jevcore PUBLIC src ${TP} ${TP}/stb)
target_compile_definitions(jevcore PRIVATE XXH_STATIC_LINKING_ONLY)
# ONNX Runtime (prebuilt, fetched by scripts/fetch-deps.sh) for CLIP.
set(ORT ${TP}/onnxruntime)
target_include_directories(jevcore PUBLIC ${ORT}/include)
target_link_libraries(jevcore PUBLIC sqlite3 ${ORT}/lib/libonnxruntime.so ${CMAKE_DL_LIBS} pthread)
set(CMAKE_BUILD_RPATH ${ORT}/lib)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

add_executable(jev-photos src/main.cpp src/i18n.cpp)
target_link_libraries(jev-photos PRIVATE jevcore imgui glfw)

add_executable(jev-photos-tests tests/test_main.cpp)
target_link_libraries(jev-photos-tests PRIVATE jevcore)

set_target_properties(jev-photos jev-photos-tests PROPERTIES BUILD_RPATH ${ORT}/lib INSTALL_RPATH "$ORIGIN/../lib")
install(TARGETS jev-photos RUNTIME DESTINATION bin)
install(FILES ${ORT}/lib/libonnxruntime.so.1 ${ORT}/lib/libonnxruntime.so.1.22.0 DESTINATION lib)
install(FILES jev-photos.desktop DESTINATION share/applications)
