# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zhxfl/purine2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zhxfl/purine2

# Include any dependencies generated for this target.
include tools/CMakeFiles/nin_data_transfer.o.dir/depend.make

# Include the progress variables for this target.
include tools/CMakeFiles/nin_data_transfer.o.dir/progress.make

# Include the compile flags for this target's objects.
include tools/CMakeFiles/nin_data_transfer.o.dir/flags.make

tools/CMakeFiles/nin_data_transfer.o.dir/nin_data_transfer.cpp.o: tools/CMakeFiles/nin_data_transfer.o.dir/flags.make
tools/CMakeFiles/nin_data_transfer.o.dir/nin_data_transfer.cpp.o: tools/nin_data_transfer.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/zhxfl/purine2/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object tools/CMakeFiles/nin_data_transfer.o.dir/nin_data_transfer.cpp.o"
	cd /home/zhxfl/purine2/tools && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/nin_data_transfer.o.dir/nin_data_transfer.cpp.o -c /home/zhxfl/purine2/tools/nin_data_transfer.cpp

tools/CMakeFiles/nin_data_transfer.o.dir/nin_data_transfer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nin_data_transfer.o.dir/nin_data_transfer.cpp.i"
	cd /home/zhxfl/purine2/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/zhxfl/purine2/tools/nin_data_transfer.cpp > CMakeFiles/nin_data_transfer.o.dir/nin_data_transfer.cpp.i

tools/CMakeFiles/nin_data_transfer.o.dir/nin_data_transfer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nin_data_transfer.o.dir/nin_data_transfer.cpp.s"
	cd /home/zhxfl/purine2/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/zhxfl/purine2/tools/nin_data_transfer.cpp -o CMakeFiles/nin_data_transfer.o.dir/nin_data_transfer.cpp.s

tools/CMakeFiles/nin_data_transfer.o.dir/nin_data_transfer.cpp.o.requires:
.PHONY : tools/CMakeFiles/nin_data_transfer.o.dir/nin_data_transfer.cpp.o.requires

tools/CMakeFiles/nin_data_transfer.o.dir/nin_data_transfer.cpp.o.provides: tools/CMakeFiles/nin_data_transfer.o.dir/nin_data_transfer.cpp.o.requires
	$(MAKE) -f tools/CMakeFiles/nin_data_transfer.o.dir/build.make tools/CMakeFiles/nin_data_transfer.o.dir/nin_data_transfer.cpp.o.provides.build
.PHONY : tools/CMakeFiles/nin_data_transfer.o.dir/nin_data_transfer.cpp.o.provides

tools/CMakeFiles/nin_data_transfer.o.dir/nin_data_transfer.cpp.o.provides.build: tools/CMakeFiles/nin_data_transfer.o.dir/nin_data_transfer.cpp.o

nin_data_transfer.o: tools/CMakeFiles/nin_data_transfer.o.dir/nin_data_transfer.cpp.o
nin_data_transfer.o: tools/CMakeFiles/nin_data_transfer.o.dir/build.make
.PHONY : nin_data_transfer.o

# Rule to build all files generated by this target.
tools/CMakeFiles/nin_data_transfer.o.dir/build: nin_data_transfer.o
.PHONY : tools/CMakeFiles/nin_data_transfer.o.dir/build

tools/CMakeFiles/nin_data_transfer.o.dir/requires: tools/CMakeFiles/nin_data_transfer.o.dir/nin_data_transfer.cpp.o.requires
.PHONY : tools/CMakeFiles/nin_data_transfer.o.dir/requires

tools/CMakeFiles/nin_data_transfer.o.dir/clean:
	cd /home/zhxfl/purine2/tools && $(CMAKE_COMMAND) -P CMakeFiles/nin_data_transfer.o.dir/cmake_clean.cmake
.PHONY : tools/CMakeFiles/nin_data_transfer.o.dir/clean

tools/CMakeFiles/nin_data_transfer.o.dir/depend:
	cd /home/zhxfl/purine2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zhxfl/purine2 /home/zhxfl/purine2/tools /home/zhxfl/purine2 /home/zhxfl/purine2/tools /home/zhxfl/purine2/tools/CMakeFiles/nin_data_transfer.o.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/CMakeFiles/nin_data_transfer.o.dir/depend

