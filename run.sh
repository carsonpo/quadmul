nvcc -std=c++17 -O3 --relocatable-device-code=false -arch=sm_80 --expt-relaxed-constexpr -Xcompiler -fdiagnostics-show-template-tree -o run kernel.cu && ./run