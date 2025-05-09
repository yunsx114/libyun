
#include "header.hpp"


#ifdef __APPLE__

// Metal headers
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>


namespace yun
{

class MetalContext{
public:
    MTL::Device* device;
    MTL::CommandQueue* queue;
    MTL::ComputePipelineState* pipeline;
    MTL::ComputeCommandEncoder* encoder;
    MTL::CommandBuffer* cmdBuffer;
    
    MetalContext(const std::string& kernelSource, const std::string& kernelName);
    ~MetalContext();
    bool execute(size_t size);
    
    static std::string getKernelSource(const std::string& key);
    static const size_t MaxDeviceData;
    static std::unordered_map<std::string, std::string> functions_map;
};

size_t const MetalContext::MaxDeviceData = 8589934592;

std::string MetalContext::getKernelSource(const std::string& key){
    auto it = functions_map.find(key);
    LOG_FATAL(it != functions_map.end(), "Metal kernel not found" + key);
    return it->second;
}

std::unordered_map<std::string, std::string> MetalContext::functions_map = {
    {"vector_add", R"(
        #include <metal_stdlib>
        using namespace metal;
        kernel void vector_add(
            device const uchar* inA [[buffer(0)]],
            device const uchar* inB [[buffer(1)]],
            device uchar* out [[buffer(2)]],
            uint id [[thread_position_in_grid]]) {
            out[id] = clamp(out[id] + inA[id] + inB[id], 0, 255);
        })"},

    {"scale_add", R"(
        #include <metal_stdlib>
        using namespace metal;
        kernel void scale_add(
            device const uchar* src [[buffer(0)]],
            device uchar* dst [[buffer(1)]],
            constant float& coeff [[buffer(2)]],
            uint id [[thread_position_in_grid]]) {
            dst[id] = clamp(dst[id] + coeff * src[id], 0.0f, 255.0f);
        })"},
    
    {"add_scale", R"(
        #include <metal_stdlib>
        using namespace metal;
        kernel void add_scale(
            device uchar* dst [[buffer(0)]],
            constant short& scale [[buffer(1)]],
            uint id [[thread_position_in_grid]]) {
            dst[id] = clamp(dst[id] + scale, 0, 255);
        })"},

    {"weighted_add", R"(
        #include <metal_stdlib>
        using namespace metal;
        kernel void weighted_add(
            device const uchar* src1 [[buffer(0)]],
            device const uchar* src2 [[buffer(1)]],
            device uchar* dst [[buffer(2)]],
            constant float& coeff1 [[buffer(3)]],
            constant float& coeff2 [[buffer(4)]],
            uint id [[thread_position_in_grid]]) {
            float result = dst[id] + coeff1 * float(src1[id]) + coeff2 * float(src2[id]);
            dst[id] = uchar(max(0.0f, min(255.0f, result)));
        })"}
};


MetalContext::MetalContext(const std::string& kernelSource, const std::string& kernelName){
    LOG_FATAL(!kernelSource.empty(),"Empty shader function");
    LOG_FATAL(!kernelName.empty(),"Empty shader name");

    this->device = MTL::CreateSystemDefaultDevice();
    LOG_FATAL(this->device,"Cannot create device");
    
    NS::Error* error = nullptr;
    NS::String* source = NS::String::string(kernelSource.c_str(), NS::UTF8StringEncoding);
    MTL::Library* library = this->device->newLibrary(source, nullptr, &error);
    source->release();
    LOG_FATAL(library,error->localizedDescription()->utf8String());
    
    MTL::Function* kernel = library->newFunction(NS::String::string(kernelName.c_str(), NS::UTF8StringEncoding));
    this->pipeline = this->device->newComputePipelineState(kernel, &error);
    kernel->release();
    library->release();
    LOG_FATAL(this->pipeline,error->localizedDescription()->utf8String());
    
    this->queue = this->device->newCommandQueue();
    this->cmdBuffer = this->queue->commandBuffer();
    this->encoder = cmdBuffer->computeCommandEncoder();
    this->encoder->setComputePipelineState(this->pipeline);
};

bool MetalContext::execute(size_t size){
    
    MTL::Size gridSize(size, 1, 1);
    NS::UInteger maxThreads = this->pipeline->maxTotalThreadsPerThreadgroup();
    MTL::Size groupSize(maxThreads, 1, 1);
    
    encoder->dispatchThreads(gridSize, groupSize);
    encoder->endEncoding();
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();
    
    return true;
}


MTL::Buffer* getMetalBuffer(const Mat& src) {
    return static_cast<MTL::Buffer*>(src.getDeviceData());
}

MetalContext::~MetalContext(){
    if(this->queue) this->queue->release();
    if(this->pipeline) this->pipeline->release();
    if(this->device) this->device->release();
    if(this->cmdBuffer) this->cmdBuffer->release();
    if(this->encoder) this->encoder->release();
};

/********************************************************************************* Mat memory**********************************************************************************/

void Mat::getAllocteMem(size_t size) {
    LOG_FATAL((size < MetalContext::MaxDeviceData), "The memory you allocate for Metal device exceeds limit 8GB");
    LOG_FATAL((size > 0), "Can't allocate zero length memory");
    MTL::Buffer* buffer = MTL::CreateSystemDefaultDevice()->newBuffer(size, MTL::ResourceStorageModeShared);
    
    gpu_ptr = buffer;
    LOG_FATAL(gpu_ptr,"Failed to allocate memory for METAL buffer");
    cpu_ptr = reinterpret_cast<uint8_t*>(buffer->contents());
    LOG_FATAL(cpu_ptr,"Failed to allocate memory for hostdata");
}

void Mat::release() {
    if (ATOMIC_ADD(refCount,-1)==1) {
        if (gpu_ptr) {
            MTL::Buffer* buffer = static_cast<MTL::Buffer*>(gpu_ptr);
            buffer->release();
        }
        delete refCount;
    }
    cpu_ptr = nullptr;
    gpu_ptr = nullptr;
    rows = cols = 0;
    channel = cType::U8C1;
    refCount = nullptr;
}

/********************************************************************************* add functitons with metal*********************************************************************************/

void addu8_METAL(const Mat& src1, const Mat& src2, Mat& dst, size_t size) {

    std::string kernelName = "vector_add";
    std::string kernelSource = MetalContext::getKernelSource(kernelName);
    MetalContext ctx(kernelSource,kernelName);

    ctx.encoder->setBuffer(getMetalBuffer(src1), 0, 0);
    ctx.encoder->setBuffer(getMetalBuffer(src2), 0, 1);
    ctx.encoder->setBuffer(getMetalBuffer(dst), 0, 2);

    ctx.execute(size);
}

void addu8_METAL(const Mat& src1, float coeff1,Mat& dst, size_t size) {

    std::string kernelName = "scale_add";
    std::string kernelSource = MetalContext::getKernelSource(kernelName);
    MetalContext ctx(kernelSource,kernelName);
    
    ctx.encoder->setBuffer(getMetalBuffer(src1), 0, 0);
    ctx.encoder->setBuffer(getMetalBuffer(dst), 0, 1);
    ctx.encoder->setBytes(&coeff1, sizeof(float), 2);
    
    ctx.execute(size);
}

void addu8_METAL(Mat& dst, short scale, size_t size){

    std::string kernelName = "add_scale";
    std::string kernelSource = MetalContext::getKernelSource(kernelName);
    MetalContext ctx(kernelSource,kernelName);
    
    ctx.encoder->setBuffer(getMetalBuffer(dst), 0, 0);
    ctx.encoder->setBytes(&scale, sizeof(short), 1);
    
    ctx.execute(size);
}


void addu8_METAL(const Mat& src1, float coeff1, const Mat& src2, float coeff2, Mat& dst, size_t size) {
    
    std::string kernelName = "weighted_add";
    std::string kernelSource = MetalContext::getKernelSource(kernelName);
    MetalContext ctx(kernelSource,kernelName);
    
    ctx.encoder->setBuffer(getMetalBuffer(src1), 0, 0);
    ctx.encoder->setBuffer(getMetalBuffer(src2), 0, 1);
    ctx.encoder->setBuffer(getMetalBuffer(dst), 0, 2);
    ctx.encoder->setBytes(&coeff1, sizeof(float), 3);
    ctx.encoder->setBytes(&coeff2, sizeof(float), 4);
    
    ctx.execute(size);
}

}

#else

namespace yun
{


void Mat::getAllocteMem(size_t size) {
    //cpu_ptr = new unsigned char[size];
    cpu_ptr = static_cast<unsigned char*>(std::aligned_alloc(64, size));
    gpu_ptr = nullptr;
    LOG_FATAL(cpu_ptr,"Failed to allocate memory for hostdata");
}


void Mat::release() {
    if (ATOMIC_ADD(refCount,-1)==1) {
        std::free(cpu_ptr);
        delete refCount;
    }
    cpu_ptr = nullptr;
    gpu_ptr = nullptr;
    rows = cols = 0;
    channel = cType::U8C1;
    refCount = nullptr;
}

void addu8_METAL(const Mat& src1, const Mat& src2, Mat& dst, size_t size) {
    LOG_FATAL(0,"Metal cannot be used on this plateform");

}

void addu8_METAL(const Mat& src1, float coeff1,Mat& dst, size_t size) {
    LOG_FATAL(0,"Metal cannot be used on this plateform");

}

void addu8_METAL (Mat& dst, short scale, size_t size){
    LOG_FATAL(0,"Metal cannot be used on this plateform");

}

void addu8_METAL(const Mat& src1, float coeff1, const Mat& src2, float coeff2, Mat& dst, size_t size) {
    LOG_FATAL(0,"Metal cannot be used on this plateform");
}


}


#endif
