cmake_minimum_required(VERSION 3.25)
project(cpt)

set(Torch_DIR ~/software/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories("~/software/c_libs")

add_executable(cpt main.cpp
        include/func.h src/func.cpp

        include/process.h src/process.cpp
        include/utils.h src/utils.cpp

        include/convnet.h src/convnet.cpp 
        include/image_io.h src/image_io.cpp 
        include/imagefolder_dataset.h src/imagefolder_dataset.cpp

        include/rnn.h src/rnn.cpp

        include/cifar10.h src/cifar10.cpp
        include/residual_block.h src/residual_block.cpp
        include/resnet.h
        include/transform.h src/transform.cpp
        
        include/dictionary.h src/dictionary.cpp
        include/corpus.h src/corpus.cpp
        include/rnn_lm.h src/rnn_lm.cpp
        
        include/vae.h src/vae.cpp
        include/vggnet.h src/vggnet.cpp)

target_link_libraries(cpt "${TORCH_LIBRARIES}")
target_include_directories(cpt PRIVATE CUDA_INCLUDE_DIRS)


set_property(TARGET cpt PROPERTY CXX_STANDARD 17)
