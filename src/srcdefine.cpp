#include "header.hpp"

namespace yun
{

/********************************************************************************* OtherFunctioons **********************************************************************************/
inline uint8_t SaturateUchar(int value){
//    return value > 255 ? 255 : value < 0 ? 0 : value;
    return (value)&(~255)?-(value)>>31 : value;
}


void setComputeMode(ComputeMode mode){
    ComputeModeManager::setMode(mode);
}

///***********************************************************************************ComputeMode*****************************************************************************************************/


ComputeMode ComputeModeManager::currentMode = ComputeMode::AUTO;

void ComputeModeManager::setMode(ComputeMode mode) {
    currentMode = mode;
}

ComputeMode ComputeModeManager::getMode() {
    return currentMode;
}

///***********************************************************************************KERNEL*****************************************************************************************************/


Kernel::Kernel() : size(0),data(0){}

Kernel::Kernel(const std::vector<float>& vec): data(vec){
    size_t s = std::sqrt(data.size());
    LOG_FATAL(s * s == data.size(), "Kernel data size must be square (e.g., 9 for 3x3)");
    size = s;
}
Kernel::Kernel(const float* arr, size_t s): size(s){
    LOG_FATAL(arr,"Empty kernel input");
    for(int i = 0 ; i < s*s ; i++){
        data.push_back(*(arr+i));
    }
}

float& Kernel::at(size_t x, size_t y){
    return data.at(y*size+x);
}
float Kernel::at(size_t x, size_t y) const{
    return data.at(y*size+x);
}
size_t Kernel::getSize(){
    return size;
}

size_t Kernel::getSize() const{
    return size;
}

///***********************************************************************************MATEXPR*****************************************************************************************************/

MatExpr::MatExpr() : scale(0) {}

void MatExpr::addMat(const Mat& m, double coefficient) {
    if(!matrices.empty()){
        size_t cur_col = this->getCols();
        size_t cur_row = this->getRows();
        cType cur_chn = this->getChannels();
        
        size_t new_col = m.getCols();
        size_t new_row = m.getRows();
        cType new_chn = m.getChannel();
        if(cur_col != new_col || cur_row != new_row || cur_chn != new_chn)
        {
            std::cerr<<"the matrice info doesn't match"<<std::endl;
        }
    }
    matrices.push_back(m);
    coefficients.push_back(coefficient);
}

void MatExpr::addScalar(double s) {
    scale += s;
}

void MatExpr::scaleBy(double s) {
    for (auto& coeff : coefficients) {
        coeff *= s;
    }
    scale *= s;
}
inline size_t MatExpr::getRows() const{
    return this->matrices[0].getRows();
}
inline size_t MatExpr::getCols() const{
    return this->matrices[0].getCols();
}
inline cType MatExpr::getChannels() const{
    return this->matrices[0].getChannel();
}

MatExpr::operator Mat() const{
    Mat result(this->matrices[0].getRows(),this->matrices[0].getCols(),this->matrices[0].getChannel());
    LOG_ERROR(assignTo(result), "Warning:No expression is assigned to Mat",0);
    return result;
}

bool MatExpr::assignTo(Mat& dst) const {
    if (matrices.empty()) {
        return false;
    }
    size_t rows = dst.getRows();
    size_t cols = dst.getCols();
    size_t channels = (int)dst.getChannel();
    size_t size = rows*cols*channels;
    dst[0] += SaturateUchar(0);// can't delete this or will cause symbol link error I don't know why
    size_t num_of_mats = matrices.size();
    bool is_odd = num_of_mats % 2;
    if(num_of_mats >= 2){
        size_t i = 0;
        for (; i < matrices.size()-1; i+=2) {
            const Mat& m1 = matrices[i];
            const Mat& m2 = matrices[i+1];
            double coeff1 = coefficients[i];
            double coeff2 = coefficients[i+1];
            if(coeff1==1&&coeff2==1)
                add(m1,m2,dst,size);
            else
                add(m1,coeff1,m2,coeff2,dst,size);
        }
        if(is_odd)add(matrices.back(),coefficients.back(),dst,size);
    }else{
        const Mat& m = matrices[0];
        double coeff = coefficients[0];m[0];//
        add(m,coeff,dst,size);
    }
    
    if(scale){
        add(dst,scale,rows*cols*channels);
    }
    
    return true;
}

MatExpr& MatExpr::operator+=(const Mat& m) {
    addMat(m, 1.0);
    return *this;
}

MatExpr& MatExpr::operator+=(double s) {
    addScalar(s);
    return *this;
}

MatExpr& MatExpr::operator+=(const MatExpr& other) {
    for (size_t i = 0; i < other.matrices.size(); i++) {
        addMat(other.matrices[i], other.coefficients[i]);
    }
    scale += other.scale;
    return *this;
}

MatExpr& MatExpr::operator-=(const Mat& m) {
    addMat(m, -1.0);
    return *this;
}

MatExpr& MatExpr::operator-=(double s) {
    addScalar(-s);
    return *this;
}

MatExpr& MatExpr::operator-=(const MatExpr& other) {
    for (size_t i = 0; i < other.matrices.size(); i++) {
        addMat(other.matrices[i], -other.coefficients[i]);
    }
    scale -= other.scale;
    return *this;
}

MatExpr& MatExpr::operator*=(double s) {
    scaleBy(s);
    return *this;
}

Mat MatExpr::operator*=(const Kernel& kernel){
    size_t row = this->getRows();
    size_t col = this->getCols();
    cType chn = this->getChannels();
    Mat newMat(row,col,chn);
    this->assignTo(newMat);
    newMat *= kernel;
    return newMat;
}

MatExpr& MatExpr::operator/=(double s) {
    if (s == 0) throw std::runtime_error("Division by zero");
    scaleBy(1.0 / s);
    return *this;
}


MatExpr operator+(const MatExpr& lhs, const MatExpr& rhs) {
    MatExpr result = lhs;
    result += rhs;
    return result;
}

MatExpr operator-(const MatExpr& lhs, const MatExpr& rhs) {
    MatExpr result = lhs;
    result -= rhs;
    return result;
}

MatExpr operator+(const MatExpr& lhs, const Mat& rhs) {
    MatExpr result = lhs;
    result += rhs;
    return result;
}

MatExpr operator+(const MatExpr& lhs, double rhs) {
    MatExpr result = lhs;
    result += rhs;
    return result;
}

MatExpr operator+(double lhs, const MatExpr& rhs) {
    MatExpr result = rhs;
    result += lhs;
    return result;
}

MatExpr operator-(const MatExpr& lhs, const Mat& rhs) {
    MatExpr result = lhs;
    result -= rhs;
    return result;
}

MatExpr operator-(const MatExpr& lhs, double rhs) {
    MatExpr result = lhs;
    result -= rhs;
    return result;
}

MatExpr operator-(double lhs, const MatExpr& rhs) {
    MatExpr result;
    result.addScalar(lhs);
    result -= rhs;
    return result;
}

Mat operator*(const Kernel& lhs, const Mat& rhs){
    return rhs * lhs;
}

MatExpr operator*(const MatExpr& lhs, double rhs) {
    MatExpr result = lhs;
    result *= rhs;
    return result;
}

MatExpr operator*(double lhs, const MatExpr& rhs) {
    MatExpr result = rhs;
    result *= lhs;
    return result;
}

MatExpr operator/(const MatExpr& lhs, double rhs) {
    MatExpr result = lhs;
    result /= rhs;
    return result;
}

MatExpr operator*(double lhs, const Mat& rhs) {
    return rhs * lhs;
}

MatExpr operator+(double lhs, const Mat& rhs) {
    return rhs + lhs;
}

MatExpr operator-(double lhs, const Mat& rhs) {
    MatExpr expr;
    expr.addScalar(lhs);
    expr.addMat(rhs, -1.0);
    return expr;
}


///***********************************************************************************Mat*****************************************************************************************************/

Mat::Mat() : rows(0), cols(0), cpu_ptr(nullptr), gpu_ptr(nullptr), channel(cType::U8C1), refCount(new int(1)) {}

Mat::Mat(size_t r, size_t c) : rows(r), cols(c), channel(cType::U8C1), refCount(new int(1))
{
    getAllocteMem(r*c);
    memset(cpu_ptr, 0, r * c);
}

Mat::Mat(size_t r, size_t c, uint8_t value) : rows(r), cols(c), channel(cType::U8C1), refCount(new int(1))
{
    getAllocteMem(r * c);
    memset(cpu_ptr, value, r * c);
}

Mat::Mat(size_t r, size_t c, cType channel):rows(r), cols(c), channel(channel), refCount(new int(1))
{
    getAllocteMem(r*c*(int)channel);
    memset(cpu_ptr, 0, r*c*(int)channel);
}
Mat::Mat(size_t r, size_t c, cType channel,uint8_t value):rows(r), cols(c), channel(channel), refCount(new int(1))
{
    size_t size = r*c*(int)channel;
    getAllocteMem(size);
    memset(cpu_ptr, value, size);
}

Mat::Mat(const Mat& other): rows(other.rows), cols(other.cols), cpu_ptr(other.cpu_ptr), gpu_ptr(other.gpu_ptr),channel(other.channel), refCount(other.refCount)
{
    LOG_FATAL(refCount,"Copied an invalid matrix");
    ATOMIC_ADD(refCount,1);
    //(*refCount)++;
}

void Mat::copyTo(Mat& other)
{
    other.release();
    other.rows = this->getRows();
    other.cols = this->getCols();
    other.channel = this->getChannel();
    other.refCount = new int(1);
    size_t size = other.getRows()*other.getCols()*(int)other.getChannel();
    other.getAllocteMem(size);
    memcpy(other.getHostData(),this->cpu_ptr,size);
}


Mat Mat::getROI(size_t r, size_t c, size_t startPos) const{
    Mat roi(r, c, this->getChannel());
    size_t channels = (size_t)this->getChannel();
    size_t srcRowByte = this->getCols() * channels;
    size_t dstRowByte = c * channels;
    
    const uint8_t* srcData = this->getHostData() + startPos;
    uint8_t* dstData = roi.getHostData();
    LOG_FATAL(srcData,"source matrix is empty");
    LOG_FATAL(dstData,"destination matrix is empty");
    for (size_t row = 0; row < r; ++row) {
        memcpy(dstData, srcData, dstRowByte);
        srcData += srcRowByte;
        dstData += dstRowByte;
    }

    return roi;
}

Mat::~Mat() { release(); }

inline size_t& Mat::getRows(){
    return this->rows;
}
inline size_t& Mat::getCols(){
    return this->cols;
}
inline size_t Mat::getRows() const{
    return this->rows;
}
inline size_t Mat::getCols() const{
    return this->cols;
}
cType& Mat::getChannel(){
    return this->channel;
}
cType Mat::getChannel() const{
    return this->channel;
}
uint8_t* Mat::getHostData(){
    LOG_FATAL(this->cpu_ptr,"Empty hostdata");
    return this->cpu_ptr;
}
uint8_t* Mat::getHostData() const{
    LOG_FATAL(this->cpu_ptr,"Empty hostdata");
    return this->cpu_ptr;
}
uint8_t& Mat::getPixel(size_t x, size_t y){
    LOG_FATAL((x < cols && y < rows),"Index exceeds range");
    return *(cpu_ptr + (y * cols + x) * (int)channel);
}
uint8_t& Mat::getPixel(size_t x, size_t y) const{
    LOG_FATAL((x < cols && y < rows),"Index exceeds range");
    return *(cpu_ptr + (y * cols + x) * (int)channel);
}
void* Mat::getDeviceData(){
    LOG_FATAL(this->gpu_ptr,"Empty devicedata");
    return this->gpu_ptr;
}
void* Mat::getDeviceData() const{
    LOG_FATAL(this->gpu_ptr,"Empty divicedata");
    return this->gpu_ptr;
}
bool Mat::isEmpty(){
    return !(this->rows && this->cols);
}
inline uint8_t& Mat::operator[](size_t index){
    size_t max_index = rows * cols * (int)channel;
    LOG_FATAL(index < max_index, "Index out of bounds");
    return this->cpu_ptr[index];
}
inline uint8_t& Mat::operator[](size_t index) const{
    size_t max_index = rows * cols * (int)channel;
    LOG_FATAL(index < max_index, "Index out of bounds");
    return this->cpu_ptr[index];
}

Mat& Mat::operator=(const Mat& other) {
    if (this != &other) {
        release();
        
        rows = other.rows;
        cols = other.cols;
        channel = other.channel;
        cpu_ptr = other.cpu_ptr;
        gpu_ptr = other.gpu_ptr;
        refCount = other.refCount;
        //(*refCount)++;
        ATOMIC_ADD(refCount,1);
    }
    return *this;
}

Mat& Mat::operator=(const MatExpr& expr) {
    size_t row = expr.getRows();
    size_t col = expr.getCols();
    cType chn = expr.getChannels();
    Mat newMat(row,col,chn);
    expr.assignTo(newMat);
    *this = newMat;
    return *this;
}

std::ostream& operator<<(std::ostream& os, const Mat& mat) {
    
    size_t rows = mat.getRows();
    size_t cols = mat.getCols();
    if (!(mat.getCols() && mat.getRows())) {
        os << "Matrix is empty";
        return os;
    }
    uint8_t* data = mat.getHostData();
    os << "[" ;
    size_t i = 0;
    for (; i < rows-1; ++i) {
        size_t j = 0;
        for (; j < cols-1; ++j) {
            os << (int)(data[i * cols + j]) << ", ";
        }
        os << (int)(data[i * cols + j]) << ";\n ";
    }
    size_t j = 0;
    for (; j < cols-1; ++j) {
        os << (int)(data[i * cols + j]) << ", ";
    }
    os << (int)(data[i * cols + j]) << "]";
    return os;
}


MatExpr Mat::operator+(const Mat& rhs) const {
    LOG_FATAL(eqsize(*this, rhs), "Mat dimensions do not match");
    MatExpr expr;
    expr.addMat(*this, 1.0);
    expr.addMat(rhs, 1.0);
    return expr;
}

MatExpr Mat::operator-(const Mat& rhs) const {
    LOG_FATAL(eqsize(*this, rhs), "Mat dimensions do not match");
    MatExpr expr;
    expr.addMat(*this, 1.0);
    expr.addMat(rhs, -1.0);
    return expr;
}

MatExpr Mat::operator*(double s) const {
    MatExpr expr;
    expr.addMat(*this, s);
    return expr;
}

Mat Mat::operator*(const Kernel& kernel) const{
    size_t row = this->getRows();
    size_t col = this->getCols();
    cType chn = this->getChannel();
    Mat newMat(row,col,chn);
    filter2D(newMat,*this,kernel);
    return newMat;
}


MatExpr Mat::operator/(double s) const {
    LOG_FATAL(s, "Divided by zero");
    MatExpr expr;
    expr.addMat(*this, 1.0 / s);
    return expr;
}

MatExpr Mat::operator+(double s) const {
    MatExpr expr;
    expr.addMat(*this, 1.0);
    expr.addScalar(s);
    return expr;
}

MatExpr Mat::operator-(double s) const {
    MatExpr expr;
    expr.addMat(*this, 1.0);
    expr.addScalar(-s);
    return expr;
}


Mat& Mat::operator+=(const Mat& m){
    *this = *this + m;
    return *this;
}
Mat& Mat::operator+=(double s){
    *this = *this + s;
    return *this;
}
Mat& Mat::operator+=(const MatExpr& other){
    *this = *this + other;
    return *this;
}
Mat& Mat::operator-=(const Mat& m){
    *this = *this - m;
    return *this;
}
Mat& Mat::operator-=(double s){
    *this = *this - s;
    return *this;
}
Mat& Mat::operator-=(const MatExpr& other){
    *this = *this - other;
    return *this;
}
Mat& Mat::operator*=(double s){
    *this = *this * s;
    return *this;
}

Mat& Mat::operator*=(const Kernel& rhs){
    *this = *this * rhs;
    return *this;
}

Mat& Mat::operator/=(double s){
    *this = *this / s;
    return *this;
}

int Mat::getReferenceCount() const {
    return *refCount;
}


/********************************************************************************* OpenCLContext **********************************************************************************/

std::string OpenCLContext::getKernelSource(const std::string& key){
    auto it = functions_map.find(key);
    LOG_FATAL(it != functions_map.end(), "OpenCL kernel not found" + key);
    return it->second;
}

std::unordered_map<std::string, std::string> OpenCLContext::functions_map = {
    {"vector_add", R"(
        __kernel void vector_add(__global const uchar* inA, 
                                 __global const uchar* inB, 
                                 __global uchar* out) {
            int id = get_global_id(0);
            out[id] = convert_uchar_sat(out[id] + inA[id] + inB[id]);
        }
    )"},

    {"scale_add", R"(
        __kernel void scale_add(__global const uchar* src,
                                __global uchar* dst,
                                float coeff) {
            int id = get_global_id(0);
            dst[id] = convert_uchar_sat(dst[id] + coeff * src[id]);
        }
    )"},

    {"add_scale", R"(
        __kernel void add_scale(__global uchar* dst,
                                short scale) {
            int id = get_global_id(0);
            dst[id] = convert_uchar_sat(dst[id] + scale);
        }
    )"},

    {"weighted_add", R"(
        __kernel void weighted_add(__global const uchar* src1,
                                   __global const uchar* src2,
                                   __global uchar* dst,
                                   float coeff1,
                                   float coeff2) {
            int id = get_global_id(0);
            dst[id] = convert_uchar_sat(dst[id] + coeff1 * src1[id] + coeff2 * src2[id]);
        }
    )"}
};


OpenCLContext::OpenCLContext(const std::string& kernelSource, const std::string& kernelName) {
    LOG_FATAL(!kernelName.empty(), "Empty kernel name");
    LOG_FATAL(!kernelSource.empty(), "Empty kernel function");

    cl_int err;

    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "OpenCL platform error: %d\n", err); }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &this->device, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "OpenCL device error: %d\n", err); }

    this->context = clCreateContext(NULL, 1, &this->device, NULL, NULL, &err);
    LOG_FATAL(this->context, "OpenCL context error");

    this->queue = clCreateCommandQueue(this->context, this->device, 0, &err);
    LOG_FATAL(this->queue, "OpenCL queue error");

    const char* source_ptr = kernelSource.c_str();
    this->program = clCreateProgramWithSource(this->context, 1, &source_ptr, NULL, &err);
    LOG_FATAL(this->program, "OpenCL program error");

    err = clBuildProgram(this->program, 1, &this->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL build error: %d\n", err);

        size_t log_size;
        clGetProgramBuildInfo(this->program, this->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::string log(log_size, '\0');
        LOG_FATAL(!log.empty(), "Failed to allocate memory for log");
        clGetProgramBuildInfo(this->program, this->device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        fprintf(stderr, "Build log:\n%s\n", log.c_str());
    }

    this->kernel = clCreateKernel(this->program, kernelName.c_str(), &err);
    LOG_FATAL(this->kernel, "OpenCL kernel error");
}


OpenCLContext::~OpenCLContext(){
    if (this->kernel) clReleaseKernel(this->kernel);
    if (this->program) clReleaseProgram(this->program);
    if (this->queue) clReleaseCommandQueue(this->queue);
    if (this->context) clReleaseContext(this->context);
};



}
