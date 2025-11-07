bash /home/karen/Documents/cactus/cactus/build.sh
cd /home/karen/Documents/cactus/tests && rm -rf build && mkdir -p build && cd build && cmake ..
cd /home/karen/Documents/cactus/tests/build && make -j$(nproc 2>/dev/null || echo 4)
cd /home/karen/Documents/cactus/tests/build && ./test_lfm2vl_model ../../weights/lfm2-vl-350m-fp16 ../monkey-nose-muzzle-wallpaper.png