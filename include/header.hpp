
#pragma once

#include <iostream>
#include <vector>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <time.h>
#include <atomic>
#include <algorithm>


// platform detection
#if defined(_WIN32) || defined(_WIN64)
    #define YUN_PLATFORM_WINDOWS 1
#elif defined(__APPLE__)
    #include <TargetConditionals.h>
    #if TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR
        #define YUN_PLATFORM_IOS 1
    #else
        #define YUN_PLATFORM_MAC 1
    #endif
#elif defined(__ANDROID__)
    #define YUN_PLATFORM_ANDROID 1
#elif defined(__linux__)
    #define YUN_PLATFORM_LINUX 1
#else
    #define YUN_PLATFORM_UNKNOWN 1
#endif

// Enable speedup according to platforms
#if defined(YUN_PLATFORM_MAC) || defined(YUN_PLATFORM_IOS)
    #include <OpenCL/opencl.h>
    #include <arm_neon.h>
    #define ENABLED_METAL 1
    #define ENABLED_NEON 1
#else
    #include <CL/cl.h>
    #define ENABLED_METAL 0
    #define ENABLED_NEON 0
#endif

#if defined(CL_VERSION_1_0) && !defined(YUN_DISABLE_OPENCL)
    #define ENABLED_OPENCL 1
#else
    #define ENABLED_OPENCL 0
#endif

#ifdef _OPENMP
    #include <omp.h>
    #define YUN_HAS_OPENMP 1
#else
    #define YUN_HAS_OPENMP 0
#endif

#if YUN_HAS_OPENMP
    #define PARALLEL_FOR #pragma omp parallel for
    //int max_threads = omp_get_max_threads();#pragma omp parallel for num_threads(max_threads) schedule(guided)
#else
    #define PARALLEL_FOR
#endif


// Export symbols or not
#if defined(YUN_PLATFORM_WINDOWS)
    #ifdef YUN_BUILD_SHARED
        #define YUN_EXPORT __declspec(dllexport)
        #define YUN_HIDE   __declspec(dllimport)
    #else
        #define YUN_EXPORT
        #define YUN_HIDE
    #endif
#else
    #define YUN_EXPORT __attribute__((visibility("default")))
    #define YUN_HIDE   __attribute__((visibility("hidden")))
#endif


// Macros used
#define SMALL_SIZE_THRESHOLD 1000000 //388800
#define MID_SIZE_THRESHOLD 43320000

#define YUN_IMIN(a, b)  ((a) ^ (((a)^(b)) & (((a) < (b)) - 1)))
#define YUN_IMAX(a, b)  ((a) ^ (((a)^(b)) & (((a) > (b)) - 1)))

#define ATOMIC_ADD(addr,delta) __c11_atomic_fetch_add((_Atomic(int)*)(addr), delta, __ATOMIC_ACQ_REL)

#define LOG_ERROR(ptr,msg,file) do{if(!(ptr)) std::cerr<<"Error occurs in file "<<__FILE__ <<", wrong at line " <<__LINE__<<", Error reason: "<<msg<<file<<std::endl; } while(0);
#define LOG_FATAL(ptr,msg) do { if(!(ptr)) throw std::runtime_error(msg); } while(0);

#define GBlur 9

#define PNG 10
#define JPG 11
#define JPEG 12
#define BMP 13

namespace yun
{

enum class YUN_EXPORT cType {
    U8C1 = 1,  // 8-bit unsigned, 1 channel (grayscale)
    U8C3 = 3,  // 8-bit unsigned, 3 channels (RGB)
    U8C4 = 4   // 8-bit unsigned, 4 channels (RGBA)
};

enum class YUN_EXPORT ComputeMode {
    AUTO=1,
    SIMD,
    METAL,
    OPENCL
};


class Mat;
class Kernel;

class YUN_EXPORT MatExpr {
public:
    MatExpr();
    //    ~MatExpr();
    void addMat(const Mat& m, double coefficient = 1.0);
    void addScalar(double s);
    void scaleBy(double s);
    
    size_t getRows() const;
    size_t getCols() const;
    cType getChannels() const;
    
    MatExpr& operator+=(const Mat& m);
    MatExpr& operator+=(double s);
    MatExpr& operator+=(const MatExpr& other);
    MatExpr& operator-=(const Mat& m);
    MatExpr& operator-=(double s);
    MatExpr& operator-=(const MatExpr& other);
    MatExpr& operator*=(double s);
    MatExpr& operator/=(double s);
    Mat operator*=(const Kernel& kernel);//
    
    
    operator Mat() const;
    bool assignTo(Mat& dst) const;
    
    YUN_EXPORT friend MatExpr operator+(const MatExpr& lhs, const MatExpr& rhs);
    YUN_EXPORT friend MatExpr operator-(const MatExpr& lhs, const MatExpr& rhs);
    YUN_EXPORT friend MatExpr operator+(const MatExpr& lhs, const Mat& rhs);
    YUN_EXPORT friend MatExpr operator+(const MatExpr& lhs, double rhs);
    YUN_EXPORT friend MatExpr operator+(double lhs, const MatExpr& rhs);
    YUN_EXPORT friend MatExpr operator-(const MatExpr& lhs, const Mat& rhs);
    YUN_EXPORT friend MatExpr operator-(const MatExpr& lhs, double rhs);
    YUN_EXPORT friend MatExpr operator-(double lhs, const MatExpr& rhs);
    YUN_EXPORT friend MatExpr operator*(const MatExpr& lhs, double rhs);
    YUN_EXPORT friend MatExpr operator*(double lhs, const MatExpr& rhs);
    YUN_EXPORT friend MatExpr operator/(const MatExpr& lhs, double rhs);
    YUN_EXPORT friend Mat operator*(const Kernel& kernel,MatExpr& rhs);//

    
private:
    std::vector<Mat> matrices;
    std::vector<double> coefficients;
    double scale;
};


class YUN_EXPORT Mat {
public:

    Mat();
    Mat(const Mat& other);
    Mat(size_t r, size_t c);
    Mat(size_t r, size_t c, uint8_t value);
    Mat(size_t r, size_t c, cType channel);
    Mat(size_t r, size_t c, cType channel,uint8_t value);
    ~Mat();
    
    size_t& getRows();
    size_t& getCols();
    cType& getChannel();
    size_t getRows() const;
    size_t getCols() const;
    cType getChannel() const;
    uint8_t*  getHostData();
    uint8_t*  getHostData() const;
    uint8_t&  getPixel(size_t row, size_t col);
    uint8_t&  getPixel(size_t row, size_t col) const;
    void*  getDeviceData();
    void*  getDeviceData() const;
    bool isEmpty();
    uint8_t& operator[](size_t index);
    uint8_t& operator[](size_t index) const;
    
    void copyTo(Mat& other);
    Mat getROI(size_t r, size_t c, size_t startPos) const;
    Mat& operator=(const Mat& other);
    Mat& operator=(const MatExpr& expr);
    
    MatExpr operator+(const Mat& rhs) const;
    MatExpr operator-(const Mat& rhs) const;
    MatExpr operator*(double s) const;
    MatExpr operator/(double s) const;
    MatExpr operator+(double s) const;
    MatExpr operator-(double s) const;
    Mat operator*(const Kernel& rhs) const;
    
    Mat& operator+=(const Mat& m);
    Mat& operator+=(double s);
    Mat& operator+=(const MatExpr& other);
    Mat& operator-=(const Mat& m);
    Mat& operator-=(double s);
    Mat& operator-=(const MatExpr& other);
    Mat& operator*=(double s);
    Mat& operator*=(const Kernel& rhs);
    Mat& operator/=(double s);
    
    YUN_EXPORT friend Mat operator*(const Kernel& lhs, const Mat& rhs);
    YUN_EXPORT friend MatExpr operator*(double lhs, const Mat& rhs);
    YUN_EXPORT friend MatExpr operator+(double lhs, const Mat& rhs);
    YUN_EXPORT friend MatExpr operator-(double lhs, const Mat& rhs);
    YUN_EXPORT friend std::ostream& operator<<(std::ostream& os, const Mat& mat);

    int getReferenceCount() const;
private:
    
    void getAllocteMem(size_t size);
    void release();
    
    uint8_t* cpu_ptr;
    void*  gpu_ptr;
    cType channel;
    size_t rows;
    size_t cols;
    int* refCount;
    
};


class YUN_EXPORT ComputeModeManager {
public:
    static ComputeMode getMode();
    static void setMode(ComputeMode mode);

private:
    static ComputeMode currentMode;
};


class YUN_EXPORT Kernel{
public:
    Kernel();
    Kernel(const std::vector<float>& vec);
    Kernel(const float* arr, size_t size = 1);
    size_t getSize();
    size_t getSize() const;
    float& at(size_t x, size_t y);
    float at(size_t x, size_t y) const;
    
private:
    size_t size;
    std::vector<float> data;
};


class OpenCLContext {
public:
    
    OpenCLContext(const std::string& kernelSource, const std::string& kernelName);
    ~OpenCLContext();
    
    static std::string getKernelSource(const std::string& key);
    static std::unordered_map<std::string, std::string> functions_map;
    
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
};



/** speed up arithmetic */
YUN_EXPORT uint8_t SaturateUchar(int value);
YUN_EXPORT void setComputeMode(ComputeMode mode);

// vector add dst[i] = src1[i] + src2[i];
void addu8(const Mat& src1, const Mat& src2, Mat& dst, size_t size);
void addu8_SIMD(const Mat& src1, const Mat& src2, Mat& dst, size_t size);
void addu8_METAL (const Mat& src1, const Mat& src2, Mat& dst, size_t size);
void addu8_OPENCL (const Mat& src1, const Mat& src2, Mat& dst, size_t size);
void add (const Mat& src1, const Mat& src2, Mat& dst, size_t size, ComputeMode mode = ComputeModeManager::getMode());

// scalar add dst[i] += coeff * src1[i];
void addu8(const Mat& src1, float coeff1, Mat& dst, size_t size);
void addu8_SIMD(const Mat& src1, float coeff1, Mat& dst, size_t size);
void addu8_METAL (const Mat& src1, float coeff1, Mat& dst, size_t size);
void addu8_OPENCL (const Mat& src1, float coeff1, Mat& dst, size_t size);
void add (const Mat& src1, float coeff1, Mat& dst, size_t size, ComputeMode mode = ComputeModeManager::getMode());

// add scalar dst[i] += s;
void addu8(Mat& dst, short scale, size_t size);
void addu8_SIMD(Mat& dst, short scale, size_t size);
void addu8_METAL (Mat& dst, short scale, size_t size);
void addu8_OPENCL (Mat& dst, short scale, size_t size);
void add (Mat& dst, short scale, size_t size, ComputeMode mode = ComputeModeManager::getMode());

// linearly add dst[i] = coeff * src1[i] + coeff2 * src2[i];
void addu8(const Mat& src1, float coeff1, const Mat& src2, float coeff2, Mat& dst, size_t size);
void addu8_SIMD(const Mat& src1, float coeff1, const Mat& src2, float coeff2, Mat& dst, size_t size);
void addu8_METAL (const Mat& src1, float coeff1, const Mat& src2, float coeff2, Mat& dst, size_t size);
void addu8_OPENCL (const Mat& src1, float coeff1, const Mat& src2, float coeff2, Mat& dst, size_t size);
void add (const Mat& src1, float coeff1, const Mat& src2, float coeff2, Mat& dst, size_t size, ComputeMode mode = ComputeModeManager::getMode());

template <typename... Args>
void add_impl(ComputeMode mode, Args&&... args);


/** image read in and write out */
YUN_EXPORT Mat imread(const std::string& filename);
YUN_EXPORT void imwrite(Mat& img, int type, const std::string& filename, int quality = 90);
bool saveImage(Mat& img, const std::string& filename, int type, int quality);


/** image process */
bool eqsize(const Mat& src1, const Mat& src2);

YUN_EXPORT void blendLinear(Mat& dst, const Mat& src1, float coeff1, const Mat& src2, float coeff2);
YUN_EXPORT void addLight(Mat& src, int scale);
YUN_EXPORT void addLight(Mat& src, float coeff);

YUN_EXPORT void generate_G_kernel_1D(float* kernel, size_t size);
YUN_EXPORT void FastGBlur(Mat& dst, const Mat& src,size_t size = 3);
YUN_EXPORT void filter2D(Mat& dst, const Mat& src, const Kernel& kernel);
YUN_EXPORT void filter2D(Mat& dst, const Mat& src, int type,size_t size = 3);


YUN_EXPORT Mat ROI(const Mat&src, size_t x1,size_t x2, size_t y1, size_t y2);
YUN_EXPORT Mat rescale(Mat&src, size_t new_width, size_t new_height);
YUN_EXPORT Mat scaleLarger(Mat&src, size_t new_width, size_t new_height);
YUN_EXPORT Mat scaleSmaller(Mat&src, size_t new_width, size_t new_height);

}
