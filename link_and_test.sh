bash /home/karen/Documents/cactus/cactus/build.sh
cd /home/karen/Documents/cactus/tests && rm -rf build && mkdir -p build && cd build && cmake ..
cd /home/karen/Documents/cactus/tests/build && make -j$(nproc 2>/dev/null || echo 4)
cd /home/karen/Documents/cactus/tests/build && CACTUS_CAPTURE_STDOUT=0 CACTUS_CAPTURE_FILE=lfm2vl_capture.txt ./test_lfm2vl_model ../../weights/lfm2-vl-350m-i8 ../../assets/test_monkey.png