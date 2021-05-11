
cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0063 NEW)

macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()

project(PEA)

set(CMAKE_MODULE_PATH 				"${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_CXX_VISIBILITY_PRESET 	hidden)
set(CMAKE_VERBOSE_MAKEFILE 			ON)
#set(CMAKE_CXX_FLAGS 				"${CMAKE_CXX_FLAGS} -std=c++1z -ggdb3 -fpermissive -Wno-write-strings -Wno-deprecated-declarations")
set (CMAKE_CXX_FL)

#set(Boost_NO_SYSTEM_PATHS ON)
#set(Boost_USE_STATIC_LIBS ON)

option(ENABLE_PARALLELISATION   "ENABLE_PARALLELISATION"    ON)
option(ENABLE_OPTIMISATION      "ENABLE_OPTIMISATION"       ON)

if(ENABLE_OPTIMISATION)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# debug
# Set debug if nothing else is set.
set(default_build_type "Debug")
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE ${default_build_type})
endif()

#find_package(Boost 1.69.0 REQUIRED COMPONENTS log log_setup date_time filesystem system thread program_options serialization timer)

set(OPENBLAS_USE_STATIC_LIBS ON)
find_package(OPENBLAS REQUIRED)

find_package(Eigen3 3.3.0)
include_directories(${EIGEN3_INCLUDE_DIRS})

find_package(OpenMP)

add_executable(pca
        src/cpp/pca.cpp
        src/cpp/pca.h
        src/cpp/pca_run.cpp
        )

target_include_directories(pca PUBLIC
        src/cpp/pca
        ${EIGEN3_INCLUDE_DIRS}
        ${LAPACK_INCLUDE_DIRS}
        ${OPENBLAS_INCLUDE_DIRS}
		"/usr/local/lib"
        )

if(OpenMP_CXX_FOUND)
    target_link_libraries(pca PUBLIC OpenMP::OpenMP_CXX)
endif()

target_compile_definitions(pca PRIVATE
							EIGEN_USE_BLAS=1
							DEBUGLOM
						  )

#==================================================
target_link_libraries(pca PUBLIC
                        m
                        pthread
                        ${OpenBLAS_LIB}
                        ${LAPACK_LIBRARIES}
					 )

print_all_variables()
