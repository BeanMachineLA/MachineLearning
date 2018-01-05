#include "netwrapper.h"
#include "./vision_server/start_server.hpp"

int main(int argc, char* argv[])
{
    image_science::startVisionServer< ImageDetector >(argc, argv);
    return 0;
}
