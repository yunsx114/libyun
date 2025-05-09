#include "header.hpp"


#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
 
namespace yun {
 

Mat imread(const std::string& filename) {
    int width, height, channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    LOG_FATAL(data,"Failed to load image");
    
    cType channelType;
    switch (channels) {
        case 1: channelType = cType::U8C1; break;
        case 3: channelType = cType::U8C3; break;
        case 4: channelType = cType::U8C4; break;
        default:
            stbi_image_free(data);
            throw std::runtime_error("Unsupported number of channels: " + std::to_string(channels));
    }
    
    Mat result(height, width, channelType);
    memcpy(result.getHostData(), data, width * height * channels);
    stbi_image_free(data);
    return result;
}


bool saveImage(Mat& img, const std::string& filename, int type, int quality) {
    int channels;
    switch (img.getChannel()) {
        case cType::U8C1: channels = 1; break;
        case cType::U8C3: channels = 3; break;
        case cType::U8C4: channels = 4; break;
        default: return false;
    }

    int success = 0;
    int row = (int)img.getRows();
    int col = (int)img.getCols();
    switch (type) {
        case PNG:
            success = stbi_write_png(filename.c_str(), col, row, channels, img.getHostData(), col * channels);
            break;
        case JPG:
        case JPEG:
            success = stbi_write_jpg(filename.c_str(), col, row, channels, img.getHostData(), quality);
            break;
        case BMP:
            success = stbi_write_bmp(filename.c_str(), col, row, channels, img.getHostData());
            break;
        default:
            return false;
    }
    return success != 0;
}

void imwrite(Mat& img, int type, const std::string& filename, int quality) {
    int succeed = saveImage(img, filename, type, quality);
    LOG_ERROR(succeed,"Failed to save image",filename);
}

}
