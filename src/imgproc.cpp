#include "header.hpp"

namespace yun
{ 

bool eqsize(const Mat& src1, const Mat& src2){
    if (src1.getRows() != src2.getRows() || src1.getCols() != src2.getCols() || src1.getChannel() != src2.getChannel()) {
        return false;
    }
    return true;
}

void blendLinear(Mat& dst, const Mat& src1, float coeff1, const Mat& src2, float coeff2){
    dst = src1 * coeff1 + src2 * coeff2;
}

void addLight(Mat& src, int scale){
    src = src + scale;
}

void addLight(Mat& src, float alpha){
    LOG_ERROR(alpha<-1.0f||alpha>1.0f,"Warning: ratio exceed limit, clamp to edge value","");
    alpha = std::clamp(alpha, -1.0f, 1.0f);
    if(alpha > 0){
        src = src * (1-alpha) + 255 * alpha;
    }else{
        src = src * (1+alpha);
    }
}

Mat ROI(const Mat&src, size_t x1,size_t y1, size_t x2, size_t y2){
    size_t row = src.getRows();
    size_t col = src.getCols();
    cType chn = src.getChannel();
    size_t newRow = y2-y1;
    size_t newCol = x2-x1;
    if(x1<0||x2<0||y1<0||y2<0||x1>=col||x2>=col||y1>=row||y2>=row||newRow <= 0 || newCol <= 0){
        LOG_ERROR(nullptr, "Invalid ROI area", 0);
        return Mat();
    }
    int startPos = (y1*(int)col+x1)*(int)chn;
    return src.getROI(newRow,newCol,startPos);
}

Mat rescale(Mat&src, size_t new_width, size_t new_height){
    size_t old_width = src.getCols();
    size_t old_height = src.getRows();
    
    if(old_width >= new_width && old_height >= new_height){
        return scaleSmaller(src,new_width,new_height);
    }else if(old_width < new_width && old_height < new_height){
        return scaleLarger(src,new_width,new_height);
    }else if(old_width >= new_width && old_height < new_height){
        Mat tempt = scaleSmaller(src,new_width,old_height);
        return scaleLarger(tempt,new_width,new_height);
    }else{// if(old_width < new_width && old_height >= new_height)
        Mat tempt = scaleSmaller(src,old_width,new_height);
        return scaleLarger(tempt,new_width,new_height);
    }
    
};

Mat scaleLarger(Mat& src, size_t new_width, size_t new_height) {
    size_t old_width = src.getCols();
    size_t old_height = src.getRows();
    int c = (int)src.getChannel();

    Mat dst(new_height, new_width, src.getChannel());

    double ratio_x = (double)old_width / new_width;
    double ratio_y = (double)old_height / new_height;

    PARALLEL_FOR
    for (size_t y = 0; y < new_height; y++) {
        double _y = y * ratio_y;
        size_t y1 = (size_t)_y;
        size_t y2 = YUN_IMIN(y1 + 1, old_height - 1);

        unsigned char* cur_y1_row = src.getHostData() + y1 * old_width * c;
        unsigned char* cur_y2_row = src.getHostData() + y2 * old_width * c;
        unsigned char* cur_new_row = dst.getHostData() + y * new_width * c;

        for (size_t x = 0; x < new_width; x++) {
            double _x = x * ratio_x;
            size_t x1 = (size_t)_x;
            size_t x2 = YUN_IMIN(x1 + 1, old_width - 1);

            float offset_x = _x - x1;
            float offset_y = _y - y1;

            size_t _x1 = x1 * c;
            size_t _x2 = x2 * c;

            for (int i = 0; i < c; i++) {
                float t1 = cur_y1_row[_x1 + i] + offset_x * (cur_y1_row[_x2 + i] - cur_y1_row[_x1 + i]);
                float t2 = cur_y2_row[_x1 + i] + offset_x * (cur_y2_row[_x2 + i] - cur_y2_row[_x1 + i]);
                cur_new_row[x * c + i] = (unsigned char)(t1 + offset_y * (t2 - t1));
            }
        }
    }

    return dst;
}

Mat scaleSmaller(Mat& src, size_t new_width, size_t new_height) {
    size_t old_width = src.getCols();
    size_t old_height = src.getRows();
    int c = (int)src.getChannel();

    double scale_x = (double)old_width / new_width;
    double scale_y = (double)old_height / new_height;

    Mat dst(new_height, new_width, src.getChannel());

    PARALLEL_FOR
    for (size_t y = 0; y < new_height; y++) {
        unsigned char* new_row = dst.getHostData() + y * new_width * c;
        for (size_t x = 0; x < new_width; x++) {
            double center_x = x * scale_x + (scale_x - 1.0) * 0.5;
            double center_y = y * scale_y + (scale_y - 1.0) * 0.5;

            size_t x1 = std::max(0.0, center_x - scale_x * 0.5);
            size_t x2 = std::min((double)(old_width - 1), center_x + scale_x * 0.5);
            size_t y1 = std::max(0.0, center_y - scale_y * 0.5);
            size_t y2 = std::min((double)(old_height - 1),center_y + scale_y * 0.5);

            double sum_c[4] = {0.0};
            size_t total_weight = (x2 - x1 + 1) * (y2 - y1 + 1);

            for (size_t _y = y1; _y <= y2; _y++) {
                const unsigned char* old_row = src.getHostData() + _y * old_width * c;
                for (size_t _x = x1; _x <= x2; _x++) {
                    for (int i = 0; i < c; i++) {
                        sum_c[i] += old_row[_x * c + i];
                    }
                }
            }

            for (int i = 0; i < c; i++) {
                new_row[x * c + i] = (unsigned char)(sum_c[i] / total_weight);
            }
        }
    }
    
    return dst;
}

void generate_G_kernel_1D(float* kernel, size_t size) {
    const int center = (int)size / 2;
    float sigma =  0.3f * ((size - 1) * 0.5f - 1) + 0.8f;
    float sum = 0.0f;
    
    for (size_t i = 0; i < size; ++i) {
        int x = i - center;
        kernel[i] = expf(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    for (size_t i = 0; i < size; ++i) kernel[i] /= sum;
}


void FastGBlur(Mat& dst, const Mat& src, size_t size) {
    if (!eqsize(dst, src)) return;

    int range = (int)size / 2;
    int channels = (int)src.getChannel();

    float* kernel_1D = (float*)malloc(size * sizeof(float));
    LOG_FATAL(kernel_1D, "Failed to allocate memory for kernel");
    generate_G_kernel_1D(kernel_1D, size);

    size_t old_width = src.getCols();
    size_t old_height = src.getRows();
    size_t actual_row_size = old_width * channels;

    unsigned char* cur_row = src.getHostData();
    unsigned char* new_row = dst.getHostData();

    PARALLEL_FOR
    for (size_t y = 0; y < old_height; y++) {
        for (size_t x = 0; x < old_width; x++) {
            float sum_c[4] = {0};

            for (int k = -range; k <= range; k++) {
                int xk = std::min(std::max(x + k, size_t(0)), old_width - size_t(1));
                size_t idx = y * actual_row_size + xk * channels;
                float w = kernel_1D[k + range];

                for (int i = 0; i < channels; i++) {
                    sum_c[i] += cur_row[idx + i] * w;
                }
            }

            size_t _idx = y * actual_row_size + x * channels;
            for (int i = 0; i < channels; i++) {
                new_row[_idx + i] = SaturateUchar(sum_c[i]);
            }
        }
    }

    PARALLEL_FOR
    for (size_t x = 0; x < old_width; x++) {
        for (size_t y = 0; y < old_height; y++) {
            float sum_c[4] = {0};

            for (int k = -range; k <= range; k++) {
                int yk = std::min(std::max(y + k, size_t(0)), old_height - size_t(1));
                size_t idx = yk * actual_row_size + x * channels;
                float w = kernel_1D[k + range];

                for (int i = 0; i < channels; i++) {
                    sum_c[i] += new_row[idx + i] * w;
                }
            }

            size_t _idx = y * actual_row_size + x * channels;
            for (int i = 0; i < channels; i++) {
                new_row[_idx + i] = SaturateUchar(sum_c[i]);
            }
        }
    }

    free(kernel_1D);
}


void filter2D(Mat& dst, const Mat& src, int type,size_t size){
    switch (type) {
        case GBlur:
            FastGBlur(dst, src, size);
            break;
        default:
            break;
    }
}


void filter2D(Mat& dst, const Mat& src, const Kernel& kernel) {
    if (!eqsize(dst, src)) return;

    int range = (int)kernel.getSize() / 2;
    size_t old_width = src.getCols();
    size_t old_height = src.getRows();
    int channels = (int)src.getChannel();
    size_t actual_row_size = old_width * channels;
    
    const unsigned char* cur_row = src.getHostData();
    unsigned char* new_row = dst.getHostData();

    PARALLEL_FOR
    for (size_t y = 0; y < old_height; y++) {
        for (size_t x = 0; x < old_width; x++) {
            float sum_c[4] = {0};
            float weight_sum = 0;

            for (int ky = -range; ky <= range; ky++) {
                int yk = std::min(std::max(y + ky, size_t(0)), old_height - size_t(1));

                for (int kx = -range; kx <= range; kx++) {
                    int xk = std::min(std::max(x + kx, size_t(0)), old_width - size_t(1));

                    size_t idx = yk * actual_row_size + xk * channels;
                    float w = kernel.at(ky + range, kx + range);

                    for (int i = 0; i < channels; i++) {
                        sum_c[i] += cur_row[idx + i] * w;
                    }
                    weight_sum += w;
                }
            }

            if(weight_sum!=0){
                const size_t _idx = y * actual_row_size + x * channels;
                for (int i = 0; i < channels; i++) {
                    new_row[_idx + i] = SaturateUchar(sum_c[i]/weight_sum);
                }
            }
        }
    }
}

}
