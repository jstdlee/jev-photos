cmake_minimum_required(VERSION 3.16)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION JEV_VERSION_FILE)
string(STRIP "${JEV_VERSION_FILE}" JEV_VERSION_DEFAULT)
set(JEV_VERSION "${JEV_VERSION_DEFAULT}" CACHE STRING "version shown by --version and in the window (CI adds the build number)")
if(WIN32)
  project(jev-photos CXX C RC)
else()
  project(jev-photos CXX C)
endif()

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

if(WIN32)
  # Windows 10+, UTF-8 everywhere (see win/jev-photos.manifest), static C++ runtime: the .exe needs no MinGW DLLs.
  add_compile_definitions(_WIN32_WINNT=0x0A00 WINVER=0x0A00 NOMINMAX _FILE_OFFSET_BITS=64)
  set(JEV_STATIC_RUNTIME -static -static-libgcc -static-libstdc++)
  set(ORT ${TP}/onnxruntime-win)
  set(ORT_LINK ${ORT}/lib/onnxruntime.dll)  # GNU ld links against the DLL directly (only OrtGetApiBase is imported)
  set(OS_LIBS ws2_32 ole32 shell32 uuid comdlg32)
else()
  # libstdc++ inside the program: the release runs on distributions older than the build machine.
  set(JEV_STATIC_RUNTIME -static-libgcc -static-libstdc++)
  set(ORT ${TP}/onnxruntime)
  set(ORT_LINK ${ORT}/lib/libonnxruntime.so)
  set(OS_LIBS ${CMAKE_DL_LIBS} pthread)
endif()

# Everything except the UI, shared by the app and the tests.
add_library(jevcore STATIC
  src/os_win.cpp src/util.cpp src/config.cpp src/http.cpp src/meta.cpp src/genmeta.cpp src/dates.cpp src/location.cpp
  src/jev.cpp src/vision.cpp src/thumbs.cpp src/clip.cpp src/llm.cpp src/search.cpp src/db.cpp src/dupes.cpp src/pipeline.cpp)
target_include_directories(jevcore PUBLIC src ${TP} ${TP}/stb)
target_compile_definitions(jevcore PRIVATE XXH_STATIC_LINKING_ONLY)
target_compile_definitions(jevcore PUBLIC JEV_VERSION="${JEV_VERSION}")
# ONNX Runtime (prebuilt, fetched by scripts/fetch-deps.sh) for CLIP.
target_include_directories(jevcore PUBLIC ${ORT}/include)
target_link_libraries(jevcore PUBLIC sqlite3 ${ORT_LINK} ${OS_LIBS})
set(CMAKE_BUILD_RPATH ${ORT}/lib)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

# The window icon, embedded: assets/jev-photos-256.png -> a byte array header.
set(ICON_PNG ${CMAKE_SOURCE_DIR}/assets/jev-photos-256.png)
set(ICON_H ${CMAKE_BINARY_DIR}/generated/icon_png.h)
file(READ ${ICON_PNG} ICON_HEX HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," ICON_BYTES "${ICON_HEX}")
file(WRITE ${ICON_H}.tmp "// generated from assets/jev-photos-256.png\nstatic const unsigned char kIconPng[] = {${ICON_BYTES}};\n")
configure_file(${ICON_H}.tmp ${ICON_H} COPYONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${ICON_PNG})

if(WIN32)
  # GUI program (no console window); the command-line modes attach to the console they were started from.
  add_executable(jev-photos WIN32 src/main.cpp src/i18n.cpp win/jev-photos.rc)
else()
  add_executable(jev-photos src/main.cpp src/i18n.cpp)
endif()
target_include_directories(jev-photos PRIVATE ${CMAKE_BINARY_DIR}/generated)
# The icon font is linked in with .incbin (no generated source, no compile cost).
target_compile_definitions(jev-photos PRIVATE ICON_TTF="${TP}/tabler/tabler-icons.ttf")
set_property(SOURCE src/main.cpp APPEND PROPERTY OBJECT_DEPENDS ${TP}/tabler/tabler-icons.ttf)
target_link_libraries(jev-photos PRIVATE jevcore imgui glfw ${JEV_STATIC_RUNTIME})

if(WIN32)
  add_executable(jev-photos-tests tests/test_main.cpp win/manifest.rc)  # UTF-8 code page for the tests too
else()
  add_executable(jev-photos-tests tests/test_main.cpp)
endif()
target_link_libraries(jev-photos-tests PRIVATE jevcore ${JEV_STATIC_RUNTIME})
if(WIN32)  # the DLL next to the programs, so they run from the build folder
  add_custom_command(TARGET jev-photos POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ORT}/lib/onnxruntime.dll $<TARGET_FILE_DIR:jev-photos>)
endif()

set_target_properties(jev-photos jev-photos-tests PROPERTIES BUILD_RPATH ${ORT}/lib INSTALL_RPATH "$ORIGIN/../lib")
if(NOT WIN32)
install(TARGETS jev-photos RUNTIME DESTINATION bin)
install(FILES assets/jev-photos.svg DESTINATION share/icons/hicolor/scalable/apps)
install(FILES assets/jev-photos-256.png DESTINATION share/icons/hicolor/256x256/apps RENAME jev-photos.png)
install(FILES ${ORT}/lib/libonnxruntime.so.1 ${ORT}/lib/libonnxruntime.so.1.22.0 DESTINATION lib)
install(FILES jev-photos.desktop DESTINATION share/applications)
endif()
