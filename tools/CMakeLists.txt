project(Examples)

cmake_minimum_required(VERSION 2.8)

find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

# Function prepares name of a test executable
#    @output_name -  output variable's name
#    @filename    -  *.cpp file path
function(tool_name output_name filename)
    get_filename_component(name ${filename} NAME_WE)
    set(${output_name} ${name}${TOOL_EXT} PARENT_SCOPE)
endfunction()

file(GLOB TOOL_CPP_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")

foreach(source ${TOOL_CPP_SOURCES})
  MESSAGE( STATUS ${source} )
endforeach()

#    Build each test separately from *.cpp files
foreach(source ${TOOL_CPP_SOURCES})
    tool_name(TOOL_NAME ${source})

    add_library(${TOOL_NAME}.o OBJECT ${source})
    set(TOOL_OBJ_LIB $<TARGET_OBJECTS:${TOOL_NAME}.o>)

    add_executable(${TOOL_NAME} ${TOOL_OBJ_LIB})
    target_link_libraries(${TOOL_NAME} purine)
    target_link_libraries(${TOOL_NAME} proto)
    target_link_libraries(${TOOL_NAME}
      ${GLOG_LIBRARIES}
      ${PROTOBUF_LIBRARIES}
      ${LIBUV_LIBRARIES}
	  )

    #    output dir
    set_target_properties(${TOOL_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)

    # Targets and object libs
    set(TOOL_TARGETS ${TOOL_TARGETS} ${TOOL_NAME})
    set(TOOL_OBJ_LIBS ${TOOL_OBJ_LIBS} ${TOOL_OBJ_LIB})
endforeach()
