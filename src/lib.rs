//! In an effort to make Vulkan synchronization more accessible, this library
//! provides a simplification of core synchronization mechanisms such as
//! pipeline barriers and events.
//!
//! Rather than the complex maze of enums and bit flags in Vulkan - many
//! combinations of which are invalid or nonsensical - this library collapses
//! this to a shorter list of distinct usage types, and a couple of options
//! for handling image layouts.
//!
//! Additionally, these usage types provide an easier mapping to other graphics
//! APIs like DirectX 12.
//!
//! Use of other synchronization mechanisms such as semaphores, fences and render
//! passes are not addressed in this library at present.
//!
//! This crate targets Vulkan synchronization 2 (`VK_KHR_synchronization2`,
//! promoted in Vulkan 1.3) — barriers use `vk::*MemoryBarrier2` with
//! `vk::PipelineStageFlags2` / `vk::AccessFlags2`, and the command helpers
//! invoke the `*2` entry points via `vk::DependencyInfo`.

use ash::vk;

pub mod cmd;

/// Defines all potential resource usages
#[derive(Debug, Copy, Clone, PartialEq, Default)]
pub enum AccessType {
    /// No access. Useful primarily for initialization
    #[default]
    Nothing,

    /// Command buffer read operation as defined by `NVX_device_generated_commands`
    CommandBufferReadNVX,

    /// Read as an indirect buffer for drawing or dispatch
    IndirectBuffer,

    /// Read as an index buffer for drawing
    IndexBuffer,

    /// Read as a vertex buffer for drawing
    VertexBuffer,

    /// Read as a uniform buffer in a vertex shader
    VertexShaderReadUniformBuffer,

    /// Read as a sampled image/uniform texel buffer in a vertex shader
    VertexShaderReadSampledImageOrUniformTexelBuffer,

    /// Read as any other resource in a vertex shader
    VertexShaderReadOther,

    /// Read as a uniform buffer in a mesh shader
    MeshShaderReadUniformBuffer,

    /// Read as a sampled image/uniform texel buffer in a mesh shader
    MeshShaderReadSampledImageOrUniformTexelBuffer,

    /// Read as any other resource in a mesh shader
    MeshShaderReadOther,

    /// Read as a uniform buffer in a task shader
    TaskShaderReadUniformBuffer,

    /// Read as a sampled image/uniform texel buffer in a task shader
    TaskShaderReadSampledImageOrUniformTexelBuffer,

    /// Read as any other resource in a task shader
    TaskShaderReadOther,

    /// Read as a uniform buffer in a tessellation control shader
    TessellationControlShaderReadUniformBuffer,

    /// Read as a sampled image/uniform texel buffer in a tessellation control shader
    TessellationControlShaderReadSampledImageOrUniformTexelBuffer,

    /// Read as any other resource in a tessellation control shader
    TessellationControlShaderReadOther,

    /// Read as a uniform buffer in a tessellation evaluation shader
    TessellationEvaluationShaderReadUniformBuffer,

    /// Read as a sampled image/uniform texel buffer in a tessellation evaluation shader
    TessellationEvaluationShaderReadSampledImageOrUniformTexelBuffer,

    /// Read as any other resource in a tessellation evaluation shader
    TessellationEvaluationShaderReadOther,

    /// Read as a uniform buffer in a geometry shader
    GeometryShaderReadUniformBuffer,

    /// Read as a sampled image/uniform texel buffer in a geometry shader
    GeometryShaderReadSampledImageOrUniformTexelBuffer,

    /// Read as any other resource in a geometry shader
    GeometryShaderReadOther,

    /// Read as a uniform buffer in a fragment shader
    FragmentShaderReadUniformBuffer,

    /// Read as a sampled image/uniform texel buffer in a fragment shader
    FragmentShaderReadSampledImageOrUniformTexelBuffer,

    /// Read as an input attachment with a color format in a fragment shader
    FragmentShaderReadColorInputAttachment,

    /// Read as an input attachment with a depth/stencil format in a fragment shader
    FragmentShaderReadDepthStencilInputAttachment,

    /// Read as any other resource in a fragment shader
    FragmentShaderReadOther,

    /// Read by blending/logic operations or subpass load operations
    ColorAttachmentRead,

    /// Read by depth/stencil tests or subpass load operations
    DepthStencilAttachmentRead,

    /// Read or written as a depth/stencil attachment during rendering, or via a subpass store op
    DepthStencilAttachmentReadWrite,

    /// Read as a uniform buffer in a compute shader
    ComputeShaderReadUniformBuffer,

    /// Read as a sampled image/uniform texel buffer in a compute shader
    ComputeShaderReadSampledImageOrUniformTexelBuffer,

    /// Read as any other resource in a compute shader
    ComputeShaderReadOther,

    /// Read or written as any resource in a compute shader
    ComputeShaderReadWrite,

    /// Read as a uniform buffer in any shader
    AnyShaderReadUniformBuffer,

    /// Read as a uniform buffer in any shader, or a vertex buffer
    AnyShaderReadUniformBufferOrVertexBuffer,

    /// Read as a sampled image in any shader
    AnyShaderReadSampledImageOrUniformTexelBuffer,

    /// Read as any other resource (excluding attachments) in any shader
    AnyShaderReadOther,

    /// Read as the source of a transfer operation
    TransferRead,

    /// Read on the host
    HostRead,

    /// Read by the presentation engine (i.e. `vkQueuePresentKHR`)
    Present,

    /// Command buffer write operation as defined by `NVX_device_generated_commands`
    CommandBufferWriteNVX,

    /// Written as any resource in a vertex shader
    VertexShaderWrite,

    /// Written as any resource in a mesh shader
    MeshShaderWrite,

    /// Written as any resource in a task shader
    TaskShaderWrite,

    /// Written as any resource in a tessellation control shader
    TessellationControlShaderWrite,

    /// Written as any resource in a tessellation evaluation shader
    TessellationEvaluationShaderWrite,

    /// Written as any resource in a geometry shader
    GeometryShaderWrite,

    /// Written as any resource in a fragment shader
    FragmentShaderWrite,

    /// Written as a color attachment during rendering, or via a subpass store op
    ColorAttachmentWrite,

    /// Written as a depth/stencil attachment during rendering, or via a subpass store op
    DepthStencilAttachmentWrite,

    /// Written as a depth aspect of a depth/stencil attachment during rendering, whilst the
    /// stencil aspect is read-only. Requires `VK_KHR_maintenance2` to be enabled.
    DepthAttachmentWriteStencilReadOnly,

    /// Written as a stencil aspect of a depth/stencil attachment during rendering, whilst the
    /// depth aspect is read-only. Requires `VK_KHR_maintenance2` to be enabled.
    StencilAttachmentWriteDepthReadOnly,

    /// Written as any resource in a compute shader
    ComputeShaderWrite,

    /// Written as any resource in any shader
    AnyShaderWrite,

    /// Written as the destination of a transfer operation
    TransferWrite,

    /// Written on the host
    HostWrite,

    /// Read or written as a color attachment during rendering
    ColorAttachmentReadWrite,

    /// Covers any access - useful for debug, generally avoid for performance reasons
    General,

    /// Read as a sampled image/uniform texel buffer in a ray tracing shader
    RayTracingShaderReadSampledImageOrUniformTexelBuffer,

    /// Read as an input attachment with a color format in a ray tracing shader
    RayTracingShaderReadColorInputAttachment,

    /// Read as an input attachment with a depth/stencil format in a ray tracing shader
    RayTracingShaderReadDepthStencilInputAttachment,

    /// Read as an acceleration structure in a ray tracing shader
    RayTracingShaderReadAccelerationStructure,

    /// Read as any other resource in a ray tracing shader
    RayTracingShaderReadOther,

    /// Written as an acceleration structure during acceleration structure building
    AccelerationStructureBuildWrite,

    /// Read as an acceleration structure during acceleration structure building (e.g. a BLAS when building a TLAS)
    AccelerationStructureBuildRead,

    /// Written as a buffer during acceleration structure building (e.g. a staging buffer)
    AccelerationStructureBufferWrite,

    /// Read as the shader binding table during ray tracing
    ShaderBindingTableRead,

    /// Read as a micromap during micromap building or ray tracing
    MicromapRead,

    /// Written as a micromap during micromap building
    MicromapWrite,

    /// Read or written as a micromap during micromap building
    MicromapReadWrite,

    /// Read as a descriptor buffer in any shader
    DescriptorBufferRead,

    /// Read as a fragment shading rate attachment
    FragmentShadingRateAttachmentRead,

    /// Read as a fragment density map attachment
    FragmentDensityMapRead,

    /// Read as a color attachment with non-coherent access (advanced blending)
    ColorAttachmentReadNoncoherent,

    /// Read as the predicate buffer for conditional rendering
    ConditionalRenderingRead,

    /// Written by transform feedback
    TransformFeedbackWrite,

    /// Read as a transform feedback counter buffer (e.g. by indirect draws)
    TransformFeedbackCounterRead,

    /// Written as a transform feedback counter buffer
    TransformFeedbackCounterWrite,

    /// Read as the invocation mask image (HUAWEI)
    InvocationMaskRead,

    /// Read as the source for video decode operations
    VideoDecodeRead,

    /// Written as the destination of video decode operations
    VideoDecodeWrite,

    /// Read or written by video decode operations
    VideoDecodeReadWrite,

    /// Read as the source for video encode operations
    VideoEncodeRead,

    /// Written as the destination of video encode operations
    VideoEncodeWrite,

    /// Read or written by video encode operations
    VideoEncodeReadWrite,

    /// Read by optical flow operations
    OpticalFlowRead,

    /// Written by optical flow operations
    OpticalFlowWrite,

    /// Read or written by optical flow operations
    OpticalFlowReadWrite,

    /// Read or written as any resource in a vertex shader
    VertexShaderReadWrite,

    /// Read or written as any resource in a tessellation control shader
    TessellationControlShaderReadWrite,

    /// Read or written as any resource in a tessellation evaluation shader
    TessellationEvaluationShaderReadWrite,

    /// Read or written as any resource in a geometry shader
    GeometryShaderReadWrite,

    /// Read or written as any resource in a fragment shader
    FragmentShaderReadWrite,

    /// Read or written as any resource in a mesh shader
    MeshShaderReadWrite,

    /// Read or written as any resource in a task shader
    TaskShaderReadWrite,

    /// Read or written as any resource in any shader
    AnyShaderReadWrite,
}
impl AccessType {
    pub const fn is_write_access(&self) -> bool {
        matches!(
            self,
            AccessType::CommandBufferWriteNVX
                | AccessType::VertexShaderWrite
                | AccessType::MeshShaderWrite
                | AccessType::TaskShaderWrite
                | AccessType::TessellationControlShaderWrite
                | AccessType::TessellationEvaluationShaderWrite
                | AccessType::GeometryShaderWrite
                | AccessType::FragmentShaderWrite
                | AccessType::ColorAttachmentWrite
                | AccessType::DepthStencilAttachmentWrite
                | AccessType::DepthAttachmentWriteStencilReadOnly
                | AccessType::DepthStencilAttachmentReadWrite
                | AccessType::StencilAttachmentWriteDepthReadOnly
                | AccessType::ComputeShaderWrite
                | AccessType::ComputeShaderReadWrite
                | AccessType::AnyShaderWrite
                | AccessType::TransferWrite
                | AccessType::HostWrite
                | AccessType::ColorAttachmentReadWrite
                | AccessType::General
                | AccessType::AccelerationStructureBuildWrite
                | AccessType::AccelerationStructureBufferWrite
                | AccessType::MicromapWrite
                | AccessType::MicromapReadWrite
                | AccessType::TransformFeedbackWrite
                | AccessType::TransformFeedbackCounterWrite
                | AccessType::VideoDecodeWrite
                | AccessType::VideoDecodeReadWrite
                | AccessType::VideoEncodeWrite
                | AccessType::VideoEncodeReadWrite
                | AccessType::OpticalFlowWrite
                | AccessType::OpticalFlowReadWrite
                | AccessType::VertexShaderReadWrite
                | AccessType::TessellationControlShaderReadWrite
                | AccessType::TessellationEvaluationShaderReadWrite
                | AccessType::GeometryShaderReadWrite
                | AccessType::FragmentShaderReadWrite
                | AccessType::MeshShaderReadWrite
                | AccessType::TaskShaderReadWrite
                | AccessType::AnyShaderReadWrite
        )
    }
}

/// Defines a handful of layout options for images.
/// Rather than a list of all possible image layouts, this reduced list is
/// correlated with the access types to map to the correct Vulkan layouts.
/// `Optimal` is usually preferred.
#[derive(Debug, Copy, Clone, PartialEq, Default)]
pub enum ImageLayout {
    /// Choose the most optimal layout for each usage. Performs layout transitions as appropriate for the access.
    #[default]
    Optimal,

    /// Layout accessible by all Vulkan access types on a device - no layout transitions except for presentation
    General,

    /// Similar to `General`, but also allows presentation engines to access it - no layout transitions.
    /// Requires `VK_KHR_shared_presentable_image` to be enabled, and this can only be used for shared presentable
    /// images (i.e. single-buffered swap chains).
    GeneralAndPresentation,
}

/// Global barriers define a set of accesses on multiple resources at once.
/// If a buffer or image doesn't require a queue ownership transfer, or an image
/// doesn't require a layout transition (e.g. you're using one of the
/// `ImageLayout::General*` layouts) then a global barrier should be preferred.
///
/// Simply define the previous and next access types of resources affected.
#[derive(Debug, Default, Clone)]
pub struct GlobalBarrier<'a> {
    pub previous_accesses: &'a [AccessType],
    pub next_accesses: &'a [AccessType],
}

/// Buffer barriers should only be used when a queue family ownership transfer
/// is required - prefer global barriers at all other times.
///
/// Access types are defined in the same way as for a global memory barrier, but
/// they only affect the buffer range identified by `buffer`, `offset` and `size`,
/// rather than all resources.
///
/// `src_queue_family_index` and `dst_queue_family_index` will be passed unmodified
/// into a buffer memory barrier.
///
/// A buffer barrier defining a queue ownership transfer needs to be executed
/// twice - once by a queue in the source queue family, and then once again by a
/// queue in the destination queue family, with a semaphore guaranteeing
/// execution order between them.
#[derive(Debug, Default, Clone)]
pub struct BufferBarrier<'a> {
    pub previous_accesses: &'a [AccessType],
    pub next_accesses: &'a [AccessType],
    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
    pub buffer: vk::Buffer,
    pub offset: usize,
    pub size: usize,
}

/// Image barriers should only be used when a queue family ownership transfer
/// or an image layout transition is required - prefer global barriers at all
/// other times.
///
/// In general it is better to use image barriers with `ImageLayout::Optimal`
/// than it is to use global barriers with images using either of the
/// `ImageLayout::General*` layouts.
///
/// Access types are defined in the same way as for a global memory barrier, but
/// they only affect the image subresource range identified by `image` and
/// `range`, rather than all resources.
///
/// `src_queue_family_index`, `dst_queue_family_index`, `image`, and `range` will
/// be passed unmodified into an image memory barrier.
///
/// An image barrier defining a queue ownership transfer needs to be executed
/// twice - once by a queue in the source queue family, and then once again by a
/// queue in the destination queue family, with a semaphore guaranteeing
/// execution order between them.
///
/// If `discard_contents` is set to true, the contents of the image become
/// undefined after the barrier is executed, which can result in a performance
/// boost over attempting to preserve the contents. This is particularly useful
/// for transient images where the contents are going to be immediately overwritten.
/// A good example of when to use this is when an application re-uses a presented
/// image after acquiring the next swap chain image.
#[derive(Debug, Default, Clone)]
pub struct ImageBarrier<'a> {
    pub previous_accesses: &'a [AccessType],
    pub next_accesses: &'a [AccessType],
    pub previous_layout: ImageLayout,
    pub next_layout: ImageLayout,
    pub discard_contents: bool,
    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
    pub image: vk::Image,
    pub range: vk::ImageSubresourceRange,
}

/// Mapping function that translates a global barrier into a synchronization 2
/// `vk::MemoryBarrier2` (with stage and access masks already populated) for use
/// with `vkCmdPipelineBarrier2` / `vkCmdWaitEvents2`.
pub fn get_memory_barrier<'a>(barrier: &GlobalBarrier<'a>) -> vk::MemoryBarrier2<'a> {
    let mut memory_barrier = vk::MemoryBarrier2::default();

    for previous_access in barrier.previous_accesses {
        let previous_info = get_access_info(previous_access);

        memory_barrier.src_stage_mask |= previous_info.stage_mask;

        // Add appropriate availability operations - for writes only.
        if previous_access.is_write_access() {
            memory_barrier.src_access_mask |= previous_info.access_mask;
        }
    }

    for next_access in barrier.next_accesses {
        let next_info = get_access_info(next_access);

        memory_barrier.dst_stage_mask |= next_info.stage_mask;

        // Add visibility operations as necessary.
        // If the src access mask, this is a WAR hazard (or for some reason a "RAR"),
        // so the dst access mask can be safely zeroed as these don't need visibility.
        if memory_barrier.src_access_mask != vk::AccessFlags2::empty() {
            memory_barrier.dst_access_mask |= next_info.access_mask;
        }
    }

    memory_barrier
}

/// Mapping function that translates a buffer barrier into a synchronization 2
/// `vk::BufferMemoryBarrier2` for use with `vkCmdPipelineBarrier2` /
/// `vkCmdWaitEvents2`.
pub fn get_buffer_memory_barrier<'a>(barrier: &BufferBarrier<'a>) -> vk::BufferMemoryBarrier2<'a> {
    let mut buffer_barrier = vk::BufferMemoryBarrier2 {
        src_queue_family_index: barrier.src_queue_family_index,
        dst_queue_family_index: barrier.dst_queue_family_index,
        buffer: barrier.buffer,
        offset: barrier.offset as u64,
        size: barrier.size as u64,
        ..Default::default()
    };

    for previous_access in barrier.previous_accesses {
        let previous_info = get_access_info(previous_access);

        buffer_barrier.src_stage_mask |= previous_info.stage_mask;

        // Add appropriate availability operations - for writes only.
        if previous_access.is_write_access() {
            buffer_barrier.src_access_mask |= previous_info.access_mask;
        }
    }

    for next_access in barrier.next_accesses {
        let next_info = get_access_info(next_access);

        buffer_barrier.dst_stage_mask |= next_info.stage_mask;

        // Add visibility operations as necessary.
        // If the src access mask, this is a WAR hazard (or for some reason a "RAR"),
        // so the dst access mask can be safely zeroed as these don't need visibility.
        if buffer_barrier.src_access_mask != vk::AccessFlags2::empty() {
            buffer_barrier.dst_access_mask |= next_info.access_mask;
        }
    }

    buffer_barrier
}

/// Mapping function that translates an image barrier into a synchronization 2
/// `vk::ImageMemoryBarrier2` for use with `vkCmdPipelineBarrier2` /
/// `vkCmdWaitEvents2`.
pub fn get_image_memory_barrier<'a>(barrier: &ImageBarrier<'a>) -> vk::ImageMemoryBarrier2<'a> {
    let mut image_barrier = vk::ImageMemoryBarrier2 {
        src_queue_family_index: barrier.src_queue_family_index,
        dst_queue_family_index: barrier.dst_queue_family_index,
        image: barrier.image,
        subresource_range: barrier.range,
        ..Default::default()
    };

    for previous_access in barrier.previous_accesses {
        let previous_info = get_access_info(previous_access);

        image_barrier.src_stage_mask |= previous_info.stage_mask;

        // Add appropriate availability operations - for writes only.
        if previous_access.is_write_access() {
            image_barrier.src_access_mask |= previous_info.access_mask;
        }

        if barrier.discard_contents {
            image_barrier.old_layout = vk::ImageLayout::UNDEFINED;
        } else {
            let layout = match barrier.previous_layout {
                ImageLayout::General => {
                    if *previous_access == AccessType::Present {
                        vk::ImageLayout::PRESENT_SRC_KHR
                    } else {
                        vk::ImageLayout::GENERAL
                    }
                }
                ImageLayout::Optimal => previous_info.image_layout,
                ImageLayout::GeneralAndPresentation => {
                    unimplemented!()
                    // TODO: layout = vk::ImageLayout::VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR
                }
            };

            image_barrier.old_layout = layout;
        }
    }

    for next_access in barrier.next_accesses {
        let next_info = get_access_info(next_access);

        image_barrier.dst_stage_mask |= next_info.stage_mask;

        // Add appropriate availability operations - in all cases beccause otherwise
        // we get WAW and RAWs.
        image_barrier.dst_access_mask |= next_info.access_mask;

        let layout = match barrier.next_layout {
            ImageLayout::General => {
                if *next_access == AccessType::Present {
                    vk::ImageLayout::PRESENT_SRC_KHR
                } else {
                    vk::ImageLayout::GENERAL
                }
            }
            ImageLayout::Optimal => next_info.image_layout,
            ImageLayout::GeneralAndPresentation => {
                unimplemented!()
                // TODO: layout = vk::ImageLayout::VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR
            }
        };

        image_barrier.new_layout = layout;
    }

    image_barrier
}

pub(crate) struct AccessInfo {
    pub(crate) stage_mask: vk::PipelineStageFlags2,
    pub(crate) access_mask: vk::AccessFlags2,
    pub(crate) image_layout: vk::ImageLayout,
}

pub(crate) fn get_access_info(access_type: &AccessType) -> AccessInfo { //TODO this function wants to be const,but bitor is const unstable on rust 1.95
    match access_type {
        AccessType::Nothing => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::empty(),
            access_mask: vk::AccessFlags2::empty(),
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::CommandBufferReadNVX => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::COMMAND_PREPROCESS_NV,
            access_mask: vk::AccessFlags2::COMMAND_PREPROCESS_READ_NV,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::IndirectBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::DRAW_INDIRECT,
            access_mask: vk::AccessFlags2::INDIRECT_COMMAND_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::IndexBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::INDEX_INPUT,
            access_mask: vk::AccessFlags2::INDEX_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::VertexBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT,
            access_mask: vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::VertexShaderReadUniformBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::VERTEX_SHADER,
            access_mask: vk::AccessFlags2::UNIFORM_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::VertexShaderReadSampledImageOrUniformTexelBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::VERTEX_SHADER,
            access_mask: vk::AccessFlags2::SHADER_SAMPLED_READ,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::VertexShaderReadOther => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::VERTEX_SHADER,
            access_mask: vk::AccessFlags2::SHADER_READ,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::MeshShaderReadUniformBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::MESH_SHADER_EXT,
            access_mask: vk::AccessFlags2::UNIFORM_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::MeshShaderReadSampledImageOrUniformTexelBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::MESH_SHADER_EXT,
            access_mask: vk::AccessFlags2::SHADER_SAMPLED_READ,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::MeshShaderReadOther => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::MESH_SHADER_EXT,
            access_mask: vk::AccessFlags2::SHADER_READ,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::TaskShaderReadUniformBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::TASK_SHADER_EXT,
            access_mask: vk::AccessFlags2::UNIFORM_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::TaskShaderReadSampledImageOrUniformTexelBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::TASK_SHADER_EXT,
            access_mask: vk::AccessFlags2::SHADER_SAMPLED_READ,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::TaskShaderReadOther => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::TASK_SHADER_EXT,
            access_mask: vk::AccessFlags2::SHADER_READ,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::TessellationControlShaderReadUniformBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
            access_mask: vk::AccessFlags2::UNIFORM_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::TessellationControlShaderReadSampledImageOrUniformTexelBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
            access_mask: vk::AccessFlags2::SHADER_SAMPLED_READ,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::TessellationControlShaderReadOther => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
            access_mask: vk::AccessFlags2::SHADER_READ,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::TessellationEvaluationShaderReadUniformBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
            access_mask: vk::AccessFlags2::UNIFORM_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::TessellationEvaluationShaderReadSampledImageOrUniformTexelBuffer => {
            AccessInfo {
                stage_mask: vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
                access_mask: vk::AccessFlags2::SHADER_SAMPLED_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }
        }
        AccessType::TessellationEvaluationShaderReadOther => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
            access_mask: vk::AccessFlags2::SHADER_READ,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::GeometryShaderReadUniformBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::GEOMETRY_SHADER,
            access_mask: vk::AccessFlags2::UNIFORM_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::GeometryShaderReadSampledImageOrUniformTexelBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::GEOMETRY_SHADER,
            access_mask: vk::AccessFlags2::SHADER_SAMPLED_READ,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::GeometryShaderReadOther => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::GEOMETRY_SHADER,
            access_mask: vk::AccessFlags2::SHADER_READ,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::FragmentShaderReadUniformBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            access_mask: vk::AccessFlags2::UNIFORM_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            access_mask: vk::AccessFlags2::SHADER_SAMPLED_READ,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::FragmentShaderReadColorInputAttachment => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            access_mask: vk::AccessFlags2::INPUT_ATTACHMENT_READ,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::FragmentShaderReadDepthStencilInputAttachment => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            access_mask: vk::AccessFlags2::INPUT_ATTACHMENT_READ,
            image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        },
        AccessType::FragmentShaderReadOther => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            access_mask: vk::AccessFlags2::SHADER_READ,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::ColorAttachmentRead => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_READ,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        },
        AccessType::DepthStencilAttachmentRead => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        },
        AccessType::DepthStencilAttachmentReadWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        },
        AccessType::ComputeShaderReadUniformBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access_mask: vk::AccessFlags2::UNIFORM_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access_mask: vk::AccessFlags2::SHADER_SAMPLED_READ,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::ComputeShaderReadOther => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access_mask: vk::AccessFlags2::SHADER_READ,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::AnyShaderReadUniformBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            access_mask: vk::AccessFlags2::UNIFORM_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::AnyShaderReadUniformBufferOrVertexBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            access_mask: vk::AccessFlags2::UNIFORM_READ | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            access_mask: vk::AccessFlags2::SHADER_SAMPLED_READ,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::AnyShaderReadOther => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            access_mask: vk::AccessFlags2::SHADER_READ,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::TransferRead => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::ALL_TRANSFER,
            access_mask: vk::AccessFlags2::TRANSFER_READ,
            image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        },
        AccessType::HostRead => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::HOST,
            access_mask: vk::AccessFlags2::HOST_READ,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::Present => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::empty(),
            access_mask: vk::AccessFlags2::empty(),
            image_layout: vk::ImageLayout::PRESENT_SRC_KHR,
        },
        AccessType::CommandBufferWriteNVX => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::COMMAND_PREPROCESS_NV,
            access_mask: vk::AccessFlags2::COMMAND_PREPROCESS_WRITE_NV,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::VertexShaderWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::VERTEX_SHADER,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::MeshShaderWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::MESH_SHADER_EXT,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::TaskShaderWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::TASK_SHADER_EXT,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::TessellationControlShaderWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::TessellationEvaluationShaderWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::GeometryShaderWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::GEOMETRY_SHADER,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::FragmentShaderWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::ColorAttachmentWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        },
        AccessType::DepthStencilAttachmentWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        },
        AccessType::DepthAttachmentWriteStencilReadOnly => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            image_layout: vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
        },
        AccessType::StencilAttachmentWriteDepthReadOnly => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            image_layout: vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
        },
        AccessType::ComputeShaderWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::ComputeShaderReadWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_READ
                | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::AnyShaderWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::TransferWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::ALL_TRANSFER,
            access_mask: vk::AccessFlags2::TRANSFER_WRITE,
            image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        },
        AccessType::HostWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::HOST,
            access_mask: vk::AccessFlags2::HOST_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::ColorAttachmentReadWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_READ
                | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        },
        AccessType::General => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            access_mask: vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::RayTracingShaderReadSampledImageOrUniformTexelBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            access_mask: vk::AccessFlags2::SHADER_SAMPLED_READ,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::RayTracingShaderReadColorInputAttachment => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            access_mask: vk::AccessFlags2::INPUT_ATTACHMENT_READ,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::RayTracingShaderReadDepthStencilInputAttachment => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            access_mask: vk::AccessFlags2::INPUT_ATTACHMENT_READ,
            image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        },
        AccessType::RayTracingShaderReadAccelerationStructure => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            access_mask: vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::RayTracingShaderReadOther => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            access_mask: vk::AccessFlags2::SHADER_READ,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::AccelerationStructureBuildWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
            access_mask: vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::AccelerationStructureBuildRead => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
            access_mask: vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::AccelerationStructureBufferWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
            access_mask: vk::AccessFlags2::TRANSFER_WRITE,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::ShaderBindingTableRead => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            access_mask: vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::MicromapRead => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::MICROMAP_BUILD_EXT
                | vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
            access_mask: vk::AccessFlags2::MICROMAP_READ_EXT,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::MicromapWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::MICROMAP_BUILD_EXT,
            access_mask: vk::AccessFlags2::MICROMAP_WRITE_EXT,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::MicromapReadWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::MICROMAP_BUILD_EXT,
            access_mask: vk::AccessFlags2::MICROMAP_READ_EXT | vk::AccessFlags2::MICROMAP_WRITE_EXT,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::DescriptorBufferRead => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            access_mask: vk::AccessFlags2::DESCRIPTOR_BUFFER_READ_EXT,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::FragmentShadingRateAttachmentRead => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR,
            access_mask: vk::AccessFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR,
            image_layout: vk::ImageLayout::FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR,
        },
        AccessType::FragmentDensityMapRead => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::FRAGMENT_DENSITY_PROCESS_EXT,
            access_mask: vk::AccessFlags2::FRAGMENT_DENSITY_MAP_READ_EXT,
            image_layout: vk::ImageLayout::FRAGMENT_DENSITY_MAP_OPTIMAL_EXT,
        },
        AccessType::ColorAttachmentReadNoncoherent => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_READ_NONCOHERENT_EXT,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        },
        AccessType::ConditionalRenderingRead => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::CONDITIONAL_RENDERING_EXT,
            access_mask: vk::AccessFlags2::CONDITIONAL_RENDERING_READ_EXT,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::TransformFeedbackWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::TRANSFORM_FEEDBACK_EXT,
            access_mask: vk::AccessFlags2::TRANSFORM_FEEDBACK_WRITE_EXT,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::TransformFeedbackCounterRead => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::DRAW_INDIRECT
                | vk::PipelineStageFlags2::TRANSFORM_FEEDBACK_EXT,
            access_mask: vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_READ_EXT,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::TransformFeedbackCounterWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::TRANSFORM_FEEDBACK_EXT,
            access_mask: vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::InvocationMaskRead => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::INVOCATION_MASK_HUAWEI,
            access_mask: vk::AccessFlags2::INVOCATION_MASK_READ_HUAWEI,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::VideoDecodeRead => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
            access_mask: vk::AccessFlags2::VIDEO_DECODE_READ_KHR,
            image_layout: vk::ImageLayout::VIDEO_DECODE_SRC_KHR,
        },
        AccessType::VideoDecodeWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
            access_mask: vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
            image_layout: vk::ImageLayout::VIDEO_DECODE_DST_KHR,
        },
        AccessType::VideoDecodeReadWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
            access_mask: vk::AccessFlags2::VIDEO_DECODE_READ_KHR
                | vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
            image_layout: vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
        },
        AccessType::VideoEncodeRead => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::VIDEO_ENCODE_KHR,
            access_mask: vk::AccessFlags2::VIDEO_ENCODE_READ_KHR,
            image_layout: vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
        },
        AccessType::VideoEncodeWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::VIDEO_ENCODE_KHR,
            access_mask: vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR,
            image_layout: vk::ImageLayout::VIDEO_ENCODE_DST_KHR,
        },
        AccessType::VideoEncodeReadWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::VIDEO_ENCODE_KHR,
            access_mask: vk::AccessFlags2::VIDEO_ENCODE_READ_KHR
                | vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR,
            image_layout: vk::ImageLayout::VIDEO_ENCODE_DPB_KHR,
        },
        AccessType::OpticalFlowRead => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::OPTICAL_FLOW_NV,
            access_mask: vk::AccessFlags2::OPTICAL_FLOW_READ_NV,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::OpticalFlowWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::OPTICAL_FLOW_NV,
            access_mask: vk::AccessFlags2::OPTICAL_FLOW_WRITE_NV,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::OpticalFlowReadWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::OPTICAL_FLOW_NV,
            access_mask: vk::AccessFlags2::OPTICAL_FLOW_READ_NV
                | vk::AccessFlags2::OPTICAL_FLOW_WRITE_NV,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::VertexShaderReadWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::VERTEX_SHADER,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_READ
                | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::TessellationControlShaderReadWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_READ
                | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::TessellationEvaluationShaderReadWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_READ
                | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::GeometryShaderReadWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::GEOMETRY_SHADER,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_READ
                | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::FragmentShaderReadWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_READ
                | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::MeshShaderReadWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::MESH_SHADER_EXT,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_READ
                | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::TaskShaderReadWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::TASK_SHADER_EXT,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_READ
                | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::AnyShaderReadWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_READ
                | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
    }
}
