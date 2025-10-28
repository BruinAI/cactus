# First rebuild the cactus library with the updated Model class
cd /home/karen/Documents/cactus/cactus
./build.sh

# Then build the debug runner
cd /home/karen/Documents/cactus/tests
rm -rf build && mkdir build && cd build
cmake ..
make siglip2_debug_runner

./siglip2_debug_runner \
    --model /home/karen/Documents/cactus/weights/lfm2-vl-350m-fp16 \
    --image ../test_synthetic.png \
    --dump-output siglip2_final_features.txt \
    --dump-layers siglip2_layer_debug.txt