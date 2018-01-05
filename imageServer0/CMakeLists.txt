cmake_minimum_required(VERSION 2.8)
project( genericServer )
find_package( PkgConfig REQUIRED )
find_package( OpenCV 3 REQUIRED)
set(OpenCV_FOUND 1)
#set (DLIB_JPEG_SUPPORT_STR
#   "Disable this if you don't want to link against libjpeg" )
#include(${CMAKE_CURRENT_SOURCE_DIR}/dlib/dlib/cmake)

SET (UTILS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/vision-utilities)

#SET( CMAKE_CXX_FLAGS " -O3 -std=c++11 -DUSE_IMPORT_EXPORT -I/usr/include -I/usr/include/python2.7 -I${UTILS_PATH}/dlib_18.18 -L/usr/local/lib -L/usr/local/lib/linux/intel64/Release -L/usr/local/lib/linux/intel64" )

IF (AWS_EC2)
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAWS_EC2" )
ENDIF()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
set (INTEL True)
set (CXX icpc)
endif()

if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
set (INTEL True)
set (CC icc)
endif()

if(DEFINED INTEL)
      SET( CMAKE_CXX_FLAGS " -O3 -std=c++11 -DUSE_IMPORT_EXPORT -I/usr/include/python2.7 -I../caffe/include -L../caffe/lib -I/usr/include -I/usr/local/cuda/include -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64 -L/usr/lib/x86_64-linux-gnu -L/usr/local/lib -L/usr/local/lib/linux/intel64/Release -L/usr/local/lib/linux/intel64" )
      SET( CMAKE_C_FLAGS " -O3 -std=c++11 -DUSE_IMPORT_EXPORT -I/usr/include/python2.7 -I/usr/include -I/usr/local/cuda/include -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64 -L/usr/lib/x86_64-linux-gnu -L/usr/local/lib -L/usr/local/lib/linux/intel64/Release -L/usr/local/lib/linux/intel64" )
else (DEFINED INTEL)
      SET( CMAKE_CXX_FLAGS " -O3 -std=c++11 -DUSE_IMPORT_EXPORT -I../caffe/include -L../caffe/lib -I/usr/include -I/usr/local/cuda/include -I/usr/include/python2.7 -L/usr/lib/x86_64-linux-gnu -L/usr/local/lib -DUSE_OPENCV -L/usr/local/lib/linux/intel64/Release -L/usr/local/lib/linux/intel64" )
endif(DEFINED INTEL)

if(DEFINED INTEL)
  INCLUDE_DIRECTORIES(/usr/local/include /usr/include/python2.7 /usr/include/X11 /opt/intel/mkl/include /usr/include/jsoncpp /usr/local/include/jsoncpp ./vision_server ${UTILS_PATH})
else(DEFINED INTEL)
  INCLUDE_DIRECTORIES(/usr/local/include /usr/include/python2.7 ../caffe/include ../caffe/lib /usr/include/X11 /usr/include/jsoncpp /usr/local/include/jsoncpp /vision_server ${UTILS_PATH})
  link_directories(${CMAKE_CURRENT_SOURCE_DIR}/caffe/lib /usr/local/cuda/lib64)
endif (DEFINED INTEL)

if(DEFINED INTEL)
   SET( LIBRARY_DEPS mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 boost_thread boost_filesystem  opencv_core opencv_imgproc opencv_imgcodecs python2.7 boost_python jsoncpp boost_system curl pthread X11 caffe glog gflags)
else(DEFINED INTEL)
   SET( LIBRARY_DEPS boost_thread boost_filesystem ${OpenCV_LIBS} python2.7 boost_python jsoncpp boost_system curl pthread X11 caffe glog gflags)
endif(DEFINED INTEL)

#INCLUDE_DIRECTORIES(/usr/local/include /usr/include/jsoncpp /usr/local/include/jsoncpp  /usr/include/python2.7 ./vision_server ${UTILS_PATH})
#include(${UTILS_PATH}/dlib_18.18/dlib/cmake)

#SET( LIBRARY_DEPS aws-cpp-sdk-sqs aws-cpp-sdk-core boost_thread dlib opencv_core opencv_imgproc opencv_imgcodecs python2.7 boost_python jsoncpp boost_system curl pthread X11)

FILE( GLOB vision_server_srcs "./vision_server/*.h*" "./vision_server/*.cpp")
FILE( GLOB loadimage_srcs "./vision-utilities/loadimage.cpp" "./vision-utilities/caffeclassifier.*" "./vision-utilities/segmenter.*")
ADD_EXECUTABLE( server ${loadimage_srcs} ${vision_server_srcs} main.cpp caffeclassifier.h caffeclassifier.cpp netwrapper.h netwrapper.cpp)

target_link_libraries( server ${LIBRARY_DEPS} ) 

