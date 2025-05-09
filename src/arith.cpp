#include "header.hpp"

namespace yun
{

// normal compute
void addu8(const Mat& src1, const Mat& src2, Mat& dst, size_t size) {
    PARALLEL_FOR
    for (size_t i = 0; i < size; i++) {
        dst[i] = SaturateUchar(dst[i] + src1[i] + src2[i]);
    }
}

void addu8(const Mat& src1, float coeff1, Mat& dst, size_t size) {
    PARALLEL_FOR
    for (size_t i = 0; i < size; i++) {
        dst[i] = SaturateUchar(dst[i] + coeff1 * src1[i]);
    }
}
void addu8(Mat& dst, short scale, size_t size){
    PARALLEL_FOR
    for (size_t i = 0; i < size; i++) {
        dst[i] = SaturateUchar(dst[i] + scale);
    }
}
void addu8(const Mat& src1, float coeff1, const Mat& src2, float coeff2, Mat& dst, size_t size) {
    PARALLEL_FOR
    for (size_t i = 0; i < size; i++) {
        dst[i] = SaturateUchar(dst[i] + coeff1 * src1[i] + coeff2 * src2[i]);
    }
}

// CPU computation with SIMD
void addu8_SIMD(const Mat& src1, const Mat& src2, Mat& dst, size_t size) {
    size_t i = 0;
#ifdef ENABLED_NEON
    PARALLEL_FOR
    for (; i + 16 <= size; i += 16) {
        uint8x16_t vec_dst = vld1q_u8(&dst[i]);
        uint8x16_t vec1 = vld1q_u8(&src1[i]);
        uint8x16_t vec2 = vld1q_u8(&src2[i]);
        
        uint8x16_t result = vqaddq_u8(vec_dst, vqaddq_u8(vec1, vec2));
        vst1q_u8(&dst[i], result);
    }
#endif
    PARALLEL_FOR
    for (; i < size; i++) {
        dst[i] = SaturateUchar(dst[i] + src1[i] + src2[i]);
    }
}

void addu8_SIMD(const Mat& src1, float coeff1, Mat& dst, size_t size) {
    size_t i = 0;
#ifdef ENABLED_NEON
    float32x4_t coeff_vec = vdupq_n_f32(coeff1);
    PARALLEL_FOR
    for (; i + 16 <= size; i += 16) {
        uint8x16_t src_u8 = vld1q_u8(&src1[i]);
        uint8x16_t dst_u8 = vld1q_u8(&dst[i]);
        
        uint16x8_t src_u16_low = vmovl_u8(vget_low_u8(src_u8));
        uint16x8_t src_u16_high = vmovl_u8(vget_high_u8(src_u8));
        float32x4_t src_f32_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(src_u16_low)));
        float32x4_t src_f32_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(src_u16_low)));
        float32x4_t src_f32_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(src_u16_high)));
        float32x4_t src_f32_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(src_u16_high)));
        
        src_f32_0 = vmulq_f32(src_f32_0, coeff_vec);
        src_f32_1 = vmulq_f32(src_f32_1, coeff_vec);
        src_f32_2 = vmulq_f32(src_f32_2, coeff_vec);
        src_f32_3 = vmulq_f32(src_f32_3, coeff_vec);
        
        uint32x4_t res_u32_0 = vcvtnq_u32_f32(src_f32_0);
        uint32x4_t res_u32_1 = vcvtnq_u32_f32(src_f32_1);
        uint32x4_t res_u32_2 = vcvtnq_u32_f32(src_f32_2);
        uint32x4_t res_u32_3 = vcvtnq_u32_f32(src_f32_3);
        
        uint16x4_t res_u16_0 = vmovn_u32(res_u32_0);
        uint16x4_t res_u16_1 = vmovn_u32(res_u32_1);
        uint16x4_t res_u16_2 = vmovn_u32(res_u32_2);
        uint16x4_t res_u16_3 = vmovn_u32(res_u32_3);
        
        uint16x8_t res_u16_low = vcombine_u16(res_u16_0, res_u16_1);
        uint16x8_t res_u16_high = vcombine_u16(res_u16_2, res_u16_3);
        
        uint8x8_t res_u8_low = vqmovun_s16(res_u16_low);
        uint8x8_t res_u8_high = vqmovun_s16(res_u16_high);
        
        uint8x16_t res_u8 = vcombine_u8(res_u8_low, res_u8_high);
        
        res_u8 = vqaddq_u8(dst_u8, res_u8);
        
        vst1q_u8(&dst[i], res_u8);
    }
#endif
    PARALLEL_FOR
    for (; i < size; i++) {
        dst[i] = SaturateUchar(dst[i] + coeff1 * src1[i]);
    }
}

void addu8_SIMD(Mat& dst, short scale, size_t size) {
    size_t i = 0;
#ifdef ENABLED_NEON
    int16x8_t scale_vec = vdupq_n_s16(scale);
    PARALLEL_FOR
    for (; i + 16 <= size; i += 16) {
        uint8x16_t dst_u8 = vld1q_u8(&dst[i]);
        int16x8_t dst_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(dst_u8)));
        int16x8_t dst_high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(dst_u8)));

        dst_low = vaddq_s16(dst_low, scale_vec);
        dst_high = vaddq_s16(dst_high, scale_vec);

        uint8x8_t res_low = vqmovun_s16(dst_low);
        uint8x8_t res_high = vqmovun_s16(dst_high);
        uint8x16_t result = vcombine_u8(res_low, res_high);

        vst1q_u8(&dst[i], result);
    }
#endif
    for (; i < size; i++) {
        dst[i] = SaturateUchar(dst[i] + scale);
    }
}

void addu8_SIMD(const Mat& src1, float coeff1, const Mat& src2, float coeff2, Mat& dst, size_t size) {
    size_t i = 0;
#ifdef ENABLED_NEON
    float32x4_t coeff1_vec = vdupq_n_f32(coeff1);
    float32x4_t coeff2_vec = vdupq_n_f32(coeff2);
    
    PARALLEL_FOR
    for (; i + 16 <= size; i += 16) {
        uint8x16_t src1_u8 = vld1q_u8(&src1[i]);
        uint8x16_t src2_u8 = vld1q_u8(&src2[i]);
        
        uint16x8_t src1_u16_low = vmovl_u8(vget_low_u8(src1_u8));
        uint16x8_t src1_u16_high = vmovl_u8(vget_high_u8(src1_u8));
        float32x4_t src1_f32_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(src1_u16_low)));
        float32x4_t src1_f32_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(src1_u16_low)));
        float32x4_t src1_f32_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(src1_u16_high)));
        float32x4_t src1_f32_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(src1_u16_high)));
        
        uint16x8_t src2_u16_low = vmovl_u8(vget_low_u8(src2_u8));
        uint16x8_t src2_u16_high = vmovl_u8(vget_high_u8(src2_u8));
        float32x4_t src2_f32_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(src2_u16_low)));
        float32x4_t src2_f32_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(src2_u16_low)));
        float32x4_t src2_f32_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(src2_u16_high)));
        float32x4_t src2_f32_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(src2_u16_high)));
        
        src1_f32_0 = vmulq_f32(src1_f32_0, coeff1_vec);
        src1_f32_1 = vmulq_f32(src1_f32_1, coeff1_vec);
        src1_f32_2 = vmulq_f32(src1_f32_2, coeff1_vec);
        src1_f32_3 = vmulq_f32(src1_f32_3, coeff1_vec);
        
        src2_f32_0 = vmulq_f32(src2_f32_0, coeff2_vec);
        src2_f32_1 = vmulq_f32(src2_f32_1, coeff2_vec);
        src2_f32_2 = vmulq_f32(src2_f32_2, coeff2_vec);
        src2_f32_3 = vmulq_f32(src2_f32_3, coeff2_vec);
        
        float32x4_t res_f32_0 = vaddq_f32(src1_f32_0, src2_f32_0);
        float32x4_t res_f32_1 = vaddq_f32(src1_f32_1, src2_f32_1);
        float32x4_t res_f32_2 = vaddq_f32(src1_f32_2, src2_f32_2);
        float32x4_t res_f32_3 = vaddq_f32(src1_f32_3, src2_f32_3);
        
        uint32x4_t res_u32_0 = vcvtnq_u32_f32(res_f32_0);
        uint32x4_t res_u32_1 = vcvtnq_u32_f32(res_f32_1);
        uint32x4_t res_u32_2 = vcvtnq_u32_f32(res_f32_2);
        uint32x4_t res_u32_3 = vcvtnq_u32_f32(res_f32_3);
        
        uint16x4_t res_u16_0 = vmovn_u32(res_u32_0);
        uint16x4_t res_u16_1 = vmovn_u32(res_u32_1);
        uint16x4_t res_u16_2 = vmovn_u32(res_u32_2);
        uint16x4_t res_u16_3 = vmovn_u32(res_u32_3);
        
        uint16x8_t res_u16_low = vcombine_u16(res_u16_0, res_u16_1);
        uint16x8_t res_u16_high = vcombine_u16(res_u16_2, res_u16_3);
        
        uint8x8_t res_u8_low = vqmovun_s16(vreinterpretq_s16_u16(res_u16_low));
        uint8x8_t res_u8_high = vqmovun_s16(vreinterpretq_s16_u16(res_u16_high));
        
        vst1q_u8(&dst[i], vcombine_u8(res_u8_low, res_u8_high));
    }
#endif
    PARALLEL_FOR
    for (; i < size; i++) {
        dst[i] = SaturateUchar(coeff1 * src1[i] + coeff2 * src2[i]);
    }
}

// OpenCL computation
void addu8_OPENCL(const Mat& src1, const Mat& src2, Mat& dst, size_t size) {
    
    std::string kernelName = "vector_add";
    std::string kernelSource = OpenCLContext::getKernelSource(kernelName);
    OpenCLContext ctx(kernelSource,kernelName);
    cl_int err;
    
    cl_mem bufA = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size, src1.getHostData(), &err);
    cl_mem bufB = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size, src2.getHostData(), &err);
    cl_mem bufC = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, size, dst.getHostData(), &err);
    
    err = clSetKernelArg(ctx.kernel, 0, sizeof(cl_mem), &bufA);
    err |= clSetKernelArg(ctx.kernel, 1, sizeof(cl_mem), &bufB);
    err |= clSetKernelArg(ctx.kernel, 2, sizeof(cl_mem), &bufC);
    
    err = clEnqueueNDRangeKernel(ctx.queue, ctx.kernel, 1, NULL, &size, NULL, 0, NULL, NULL);
    
    err = clEnqueueReadBuffer(ctx.queue, bufC, CL_TRUE, 0, size, dst.getHostData(), 0, NULL, NULL);
    
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
}

void addu8_OPENCL(const Mat& src1, float coeff1, Mat& dst, size_t size) {

    std::string kernelName = "scale_add";
    std::string kernelSource = OpenCLContext::getKernelSource(kernelName);
    OpenCLContext ctx(kernelSource,kernelName);
    cl_int err;
    
    cl_mem bufSrc = clCreateBuffer(ctx.context,CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,size,src1.getHostData(),&err);
    cl_mem bufDst = clCreateBuffer(ctx.context,CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,size,dst.getHostData(),&err);
    
    err = clSetKernelArg(ctx.kernel, 0, sizeof(cl_mem), &bufSrc);
    err |= clSetKernelArg(ctx.kernel, 1, sizeof(cl_mem), &bufDst);
    err |= clSetKernelArg(ctx.kernel, 2, sizeof(float), &coeff1);
    
    size_t globalSize = size;
    err = clEnqueueNDRangeKernel(ctx.queue,ctx.kernel,1,NULL,&globalSize,NULL,0,NULL,NULL);
    err = clEnqueueReadBuffer(ctx.queue,bufDst,CL_TRUE,0,size,dst.getHostData(),0,NULL,NULL);
    
    clReleaseMemObject(bufSrc);
    clReleaseMemObject(bufDst);
}

void addu8_OPENCL (Mat& dst, short scale, size_t size){

    std::string kernelName = "add_scale";
    std::string kernelSource = OpenCLContext::getKernelSource(kernelName);
    OpenCLContext ctx(kernelSource,kernelName);
    cl_int err;
    
    cl_mem bufDst = clCreateBuffer(ctx.context,CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,size,dst.getHostData(),&err);
    
    err |= clSetKernelArg(ctx.kernel, 0, sizeof(cl_mem), &bufDst);
    err |= clSetKernelArg(ctx.kernel, 1, sizeof(short), &scale);
    
    size_t globalSize = size;
    err = clEnqueueNDRangeKernel(ctx.queue,ctx.kernel,1,NULL,&globalSize,NULL,0,NULL,NULL);
    err = clEnqueueReadBuffer(ctx.queue,bufDst,CL_TRUE,0,size,dst.getHostData(),0,NULL,NULL);
    
    clReleaseMemObject(bufDst);
    
}

void addu8_OPENCL(const Mat& src1, float coeff1, const Mat& src2, float coeff2, Mat& dst, size_t size) {

    std::string kernelName = "weighted_add";
    std::string kernelSource = OpenCLContext::getKernelSource(kernelName);
    OpenCLContext ctx(kernelSource, kernelName);
    cl_int err;
    
    cl_mem bufSrc1 = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size, src1.getHostData(), &err);
    cl_mem bufSrc2 = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size, src2.getHostData(), &err);
    cl_mem bufDst = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, size, dst.getHostData(), &err);
    
    err = clSetKernelArg(ctx.kernel, 0, sizeof(cl_mem), &bufSrc1);
    err |= clSetKernelArg(ctx.kernel, 1, sizeof(cl_mem), &bufSrc2);
    err |= clSetKernelArg(ctx.kernel, 2, sizeof(cl_mem), &bufDst);
    err |= clSetKernelArg(ctx.kernel, 3, sizeof(float), &coeff1);
    err |= clSetKernelArg(ctx.kernel, 4, sizeof(float), &coeff2);
    
    size_t globalSize = size;
    err = clEnqueueNDRangeKernel(ctx.queue,ctx.kernel,1,NULL,&globalSize,NULL,0,NULL,NULL);
    err = clEnqueueReadBuffer(ctx.queue,bufDst,CL_TRUE,0,size,dst.getHostData(),0,NULL,NULL);
    
    clReleaseMemObject(bufSrc1);
    clReleaseMemObject(bufSrc2);
    clReleaseMemObject(bufDst);
}


template <typename... Args>
void add_impl(ComputeMode mode, size_t size, Args&&... args) {
    
    if (mode == ComputeMode::AUTO) {
        if (size < SMALL_SIZE_THRESHOLD) {
            mode = ComputeMode::SIMD;
        }
        else if (size < MID_SIZE_THRESHOLD) {
#ifdef __APPLE__
            mode = ENABLED_OPENCL ? ComputeMode::OPENCL : ComputeMode::METAL;
#else
            mode = ComputeMode::OPENCL;
#endif
        }
        else {
#ifdef __APPLE__
            mode = ComputeMode::METAL;
#else
            mode = ENABLED_OPENCL ? ComputeMode::OPENCL : ComputeMode::SIMD;
#endif
        }
    }
    
    switch(mode)
    {
        case ComputeMode::SIMD:
            LOG_FATAL(ENABLED_NEON, "WARNING:NEON SIMD not supported")
            addu8_SIMD(std::forward<Args>(args)...,size);
            break;
        case ComputeMode::METAL:
            LOG_FATAL(ENABLED_METAL, "WARNING:METAL not supported")
            addu8_METAL(std::forward<Args>(args)...,size);
            break;
        case ComputeMode::OPENCL:
            LOG_FATAL(ENABLED_OPENCL, "WARNING:OPENCL not supported")
            addu8_OPENCL(std::forward<Args>(args)...,size);
            break;
        default:
            addu8(std::forward<Args>(args)...,size);
    }
    
}

void add(const Mat& src1, const Mat& src2, Mat& dst, size_t size, ComputeMode mode) {
    add_impl(mode, size, src1, src2, dst);
}

void add(const Mat& src1, float coeff1, Mat& dst, size_t size, ComputeMode mode) {
    add_impl(mode, size, src1, coeff1, dst);
}

void add(Mat& dst, short scale, size_t size, ComputeMode mode) {
    add_impl(mode, size, dst, scale);
}

void add(const Mat& src1, float coeff1, const Mat& src2, float coeff2, Mat& dst, size_t size,
         ComputeMode mode) {
    add_impl(mode, size, src1, coeff1, src2, coeff2, dst);
}


}
