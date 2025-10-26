#!/bin/bash
# SigLip2 Preprocessor Test Runner
# This script compiles and runs the C++ test and provides instructions for Python comparison

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== SigLip2 Preprocessor Test Runner ===${NC}\n"

# Check if image path is provided
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: No image path provided${NC}"
    echo "Usage: $0 <image_path>"
    echo "Example: $0 ../assets/banner.jpg"
    exit 1
fi

IMAGE_PATH="$1"
IMAGE_NAME=$(basename "$IMAGE_PATH")

# Check if image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo -e "${RED}Error: Image file not found: $IMAGE_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}Image found: $IMAGE_PATH${NC}\n"

# Step 1: Build the main cactus library
echo -e "${YELLOW}Step 1: Building cactus library...${NC}"
cd ../cactus
if [ ! -d "build" ]; then
    mkdir build
fi
cd build
cmake .. > /dev/null
make -j$(nproc) > /dev/null 2>&1 || make -j4 > /dev/null 2>&1
echo -e "${GREEN}✓ Cactus library built${NC}\n"

# Step 2: Build the tests
echo -e "${YELLOW}Step 2: Building tests...${NC}"
cd ../../tests
if [ ! -d "build" ]; then
    mkdir build
fi
cd build
cmake .. > /dev/null
make -j$(nproc) > /dev/null 2>&1 || make -j4 > /dev/null 2>&1
echo -e "${GREEN}✓ Tests built${NC}\n"

# Step 3: Run C++ test
echo -e "${YELLOW}Step 3: Running C++ preprocessor test...${NC}"
cd ..
OUTPUT_CPP="siglip2_output_cpp_${IMAGE_NAME}.txt"
./build/test_siglip2_preprocessor "$IMAGE_PATH" "$OUTPUT_CPP"
echo -e "${GREEN}✓ C++ test completed${NC}\n"

# Step 4: Instructions for Python test
echo -e "${BLUE}=== Next Steps ===${NC}"
echo -e "Run the Python reference implementation to compare outputs:\n"
echo -e "${YELLOW}Option 1 - Local Python:${NC}"
echo -e "  python3 test_siglip2_reference.py \"$IMAGE_PATH\" siglip2_output_python_${IMAGE_NAME}.txt\n"

echo -e "${YELLOW}Option 2 - Google Colab:${NC}"
echo -e "  1. Upload test_siglip2_reference.py to Colab"
echo -e "  2. Upload your image: $IMAGE_PATH"
echo -e "  3. Run: !python test_siglip2_reference.py \"$IMAGE_NAME\" siglip2_output_python.txt"
echo -e "  4. Download siglip2_output_python.txt\n"

echo -e "${BLUE}Compare outputs:${NC}"
echo -e "  diff $OUTPUT_CPP siglip2_output_python_${IMAGE_NAME}.txt\n"
echo -e "  Or use any text comparison tool to check if values match\n"

echo -e "${GREEN}Done! C++ output saved to: $OUTPUT_CPP${NC}"

