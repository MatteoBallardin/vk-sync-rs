//! Extended tests based on additional cases from the Vulkan-Docs synchronization
//! examples wiki: https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples.

use ash::vk;
use vk_sync_fork as vk_sync;

fn default_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::empty(),
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

fn make_image_barrier<'a>(
    previous: &'a [vk_sync::AccessType],
    next: &'a [vk_sync::AccessType],
    previous_layout: vk_sync::ImageLayout,
    next_layout: vk_sync::ImageLayout,
    discard_contents: bool,
) -> vk_sync::ImageBarrier<'a> {
    vk_sync::ImageBarrier {
        previous_accesses: previous,
        next_accesses: next,
        previous_layout,
        next_layout,
        discard_contents,
        src_queue_family_index: 0,
        dst_queue_family_index: 0,
        image: vk::Image::null(),
        range: default_range(),
    }
}

// ---------------------------------------------------------------------------
// Global / memory barriers
// ---------------------------------------------------------------------------

#[test]
fn host_write_compute_read_uniform() {
    // Host write to buffer, Compute read as uniform buffer.
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::HostWrite],
        next_accesses: &[vk_sync::AccessType::ComputeShaderReadUniformBuffer],
    };

    let barrier = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::HOST);
    assert_eq!(
        barrier.dst_stage_mask,
        vk::PipelineStageFlags2::COMPUTE_SHADER
    );
    assert_eq!(barrier.src_access_mask, vk::AccessFlags2::HOST_WRITE);
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags2::UNIFORM_READ);
}

#[test]
fn compute_read_compute_write_war_hazard() {
    // Compute read (no prior writes), Compute write — write-after-read hazard.
    // Only execution dependency is needed; no availability/visibility operations.
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::ComputeShaderReadOther],
        next_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
    };

    let barrier = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(
        barrier.src_stage_mask,
        vk::PipelineStageFlags2::COMPUTE_SHADER
    );
    assert_eq!(
        barrier.dst_stage_mask,
        vk::PipelineStageFlags2::COMPUTE_SHADER
    );
    assert_eq!(barrier.src_access_mask, vk::AccessFlags2::empty());
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags2::empty());
}

#[test]
fn compute_write_compute_write_waw_hazard() {
    // Compute write, Compute write — write-after-write hazard.
    // Both availability and visibility operations are required.
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
        next_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
    };

    let barrier = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(
        barrier.src_stage_mask,
        vk::PipelineStageFlags2::COMPUTE_SHADER
    );
    assert_eq!(
        barrier.dst_stage_mask,
        vk::PipelineStageFlags2::COMPUTE_SHADER
    );
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags2::SHADER_STORAGE_WRITE
    );
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags2::SHADER_STORAGE_WRITE
    );
}

#[test]
fn transfer_write_host_read() {
    // Transfer write to buffer, Host read from buffer.
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::TransferWrite],
        next_accesses: &[vk_sync::AccessType::HostRead],
    };

    let barrier = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(
        barrier.src_stage_mask,
        vk::PipelineStageFlags2::ALL_TRANSFER
    );
    assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::HOST);
    assert_eq!(barrier.src_access_mask, vk::AccessFlags2::TRANSFER_WRITE);
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags2::HOST_READ);
}

#[test]
fn compute_write_multiple_graphics_reads() {
    // Compute write, then several graphics-stage reads in parallel.
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
        next_accesses: &[
            vk_sync::AccessType::IndirectBuffer,
            vk_sync::AccessType::IndexBuffer,
            vk_sync::AccessType::VertexBuffer,
            vk_sync::AccessType::FragmentShaderReadUniformBuffer,
        ],
    };

    let barrier = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(
        barrier.src_stage_mask,
        vk::PipelineStageFlags2::COMPUTE_SHADER
    );
    assert_eq!(
        barrier.dst_stage_mask,
        vk::PipelineStageFlags2::DRAW_INDIRECT
            | vk::PipelineStageFlags2::INDEX_INPUT
            | vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT
            | vk::PipelineStageFlags2::FRAGMENT_SHADER
    );
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags2::SHADER_STORAGE_WRITE
    );
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags2::INDIRECT_COMMAND_READ
            | vk::AccessFlags2::INDEX_READ
            | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ
            | vk::AccessFlags2::UNIFORM_READ
    );
}

#[test]
fn acceleration_structure_build_ray_tracing_read() {
    // Build acceleration structure, then read it during ray tracing.
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::AccelerationStructureBuildWrite],
        next_accesses: &[vk_sync::AccessType::RayTracingShaderReadAccelerationStructure],
    };

    let barrier = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(
        barrier.src_stage_mask,
        vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR
    );
    assert_eq!(
        barrier.dst_stage_mask,
        vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
    );
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR
    );
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR
    );
}

#[test]
fn acceleration_structure_build_blas_read_for_tlas() {
    // BLAS build, followed by reading it as input when building a TLAS.
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::AccelerationStructureBuildWrite],
        next_accesses: &[vk_sync::AccessType::AccelerationStructureBuildRead],
    };

    let barrier = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(
        barrier.src_stage_mask,
        vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR
    );
    assert_eq!(
        barrier.dst_stage_mask,
        vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR
    );
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR
    );
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR
    );
}

#[test]
fn transform_feedback_write_indirect_draw_counter() {
    // Transform feedback writes a counter, indirect draw consumes it.
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::TransformFeedbackCounterWrite],
        next_accesses: &[vk_sync::AccessType::TransformFeedbackCounterRead],
    };

    let barrier = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(
        barrier.src_stage_mask,
        vk::PipelineStageFlags2::TRANSFORM_FEEDBACK_EXT
    );
    assert_eq!(
        barrier.dst_stage_mask,
        vk::PipelineStageFlags2::DRAW_INDIRECT | vk::PipelineStageFlags2::TRANSFORM_FEEDBACK_EXT
    );
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT
    );
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_READ_EXT
    );
}

#[test]
fn task_shader_write_mesh_shader_read() {
    // Task shader writes a payload, mesh shader reads it.
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::TaskShaderWrite],
        next_accesses: &[vk_sync::AccessType::MeshShaderReadOther],
    };

    let barrier = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(
        barrier.src_stage_mask,
        vk::PipelineStageFlags2::TASK_SHADER_EXT
    );
    assert_eq!(
        barrier.dst_stage_mask,
        vk::PipelineStageFlags2::MESH_SHADER_EXT
    );
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags2::SHADER_STORAGE_WRITE
    );
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags2::SHADER_READ);
}

#[test]
fn compute_read_write_chained() {
    // Compute read+write storage, then compute read+write storage again.
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::ComputeShaderReadWrite],
        next_accesses: &[vk_sync::AccessType::ComputeShaderReadWrite],
    };

    let barrier = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(
        barrier.src_stage_mask,
        vk::PipelineStageFlags2::COMPUTE_SHADER
    );
    assert_eq!(
        barrier.dst_stage_mask,
        vk::PipelineStageFlags2::COMPUTE_SHADER
    );
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE
    );
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE
    );
}

#[test]
fn nothing_to_nothing_noop() {
    // No-op barrier — nothing to nothing.
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::Nothing],
        next_accesses: &[vk_sync::AccessType::Nothing],
    };

    let barrier = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::NONE);
    assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::NONE);
    assert_eq!(barrier.src_access_mask, vk::AccessFlags2::empty());
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags2::empty());
}

// ---------------------------------------------------------------------------
// Buffer barriers (queue ownership transfer)
// ---------------------------------------------------------------------------

#[test]
fn buffer_queue_ownership_transfer() {
    // Release/acquire a buffer between two queue families.
    let buffer_barrier = vk_sync::BufferBarrier {
        previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
        next_accesses: &[vk_sync::AccessType::VertexBuffer],
        src_queue_family_index: 1,
        dst_queue_family_index: 2,
        buffer: vk::Buffer::null(),
        offset: 0,
        size: 1024,
    };

    let barrier = vk_sync::get_buffer_memory_barrier(&buffer_barrier);

    assert_eq!(barrier.src_queue_family_index, 1);
    assert_eq!(barrier.dst_queue_family_index, 2);
    assert_eq!(barrier.offset, 0);
    assert_eq!(barrier.size, 1024);
    assert_eq!(
        barrier.src_stage_mask,
        vk::PipelineStageFlags2::COMPUTE_SHADER
    );
    assert_eq!(
        barrier.dst_stage_mask,
        vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT
    );
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags2::SHADER_STORAGE_WRITE
    );
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags2::VERTEX_ATTRIBUTE_READ
    );
}

// ---------------------------------------------------------------------------
// Image barriers
// ---------------------------------------------------------------------------

#[test]
fn image_initial_layout_transition_with_discard() {
    // First use of an image — discard contents and transition to color attachment.
    let image_barrier = make_image_barrier(
        &[vk_sync::AccessType::Nothing],
        &[vk_sync::AccessType::ColorAttachmentWrite],
        vk_sync::ImageLayout::Optimal,
        vk_sync::ImageLayout::Optimal,
        true,
    );

    let barrier = vk_sync::get_image_memory_barrier(&image_barrier);

    assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::NONE);
    assert_eq!(
        barrier.dst_stage_mask,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
    );
    assert_eq!(barrier.src_access_mask, vk::AccessFlags2::empty());
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
    );
    assert_eq!(barrier.old_layout, vk::ImageLayout::UNDEFINED);
    assert_eq!(
        barrier.new_layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );
}

#[test]
fn image_present_to_color_attachment() {
    // Re-use a presented swap chain image as a color attachment.
    let image_barrier = make_image_barrier(
        &[vk_sync::AccessType::Present],
        &[vk_sync::AccessType::ColorAttachmentWrite],
        vk_sync::ImageLayout::Optimal,
        vk_sync::ImageLayout::Optimal,
        true,
    );

    let barrier = vk_sync::get_image_memory_barrier(&image_barrier);

    assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::NONE);
    assert_eq!(
        barrier.dst_stage_mask,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
    );
    assert_eq!(barrier.src_access_mask, vk::AccessFlags2::empty());
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
    );
    assert_eq!(barrier.old_layout, vk::ImageLayout::UNDEFINED);
    assert_eq!(
        barrier.new_layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );
}

#[test]
fn image_transfer_dst_to_present() {
    // Blit/copy result presented directly.
    let image_barrier = make_image_barrier(
        &[vk_sync::AccessType::TransferWrite],
        &[vk_sync::AccessType::Present],
        vk_sync::ImageLayout::Optimal,
        vk_sync::ImageLayout::Optimal,
        false,
    );

    let barrier = vk_sync::get_image_memory_barrier(&image_barrier);

    assert_eq!(
        barrier.src_stage_mask,
        vk::PipelineStageFlags2::ALL_TRANSFER
    );
    assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::NONE);
    assert_eq!(barrier.src_access_mask, vk::AccessFlags2::TRANSFER_WRITE);
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags2::empty());
    assert_eq!(barrier.old_layout, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
    assert_eq!(barrier.new_layout, vk::ImageLayout::PRESENT_SRC_KHR);
}

#[test]
fn image_general_layout_compute_pingpong() {
    // Compute write then compute read using GENERAL layout — no layout transition.
    let image_barrier = make_image_barrier(
        &[vk_sync::AccessType::ComputeShaderWrite],
        &[vk_sync::AccessType::ComputeShaderReadOther],
        vk_sync::ImageLayout::General,
        vk_sync::ImageLayout::General,
        false,
    );

    let barrier = vk_sync::get_image_memory_barrier(&image_barrier);

    assert_eq!(
        barrier.src_stage_mask,
        vk::PipelineStageFlags2::COMPUTE_SHADER
    );
    assert_eq!(
        barrier.dst_stage_mask,
        vk::PipelineStageFlags2::COMPUTE_SHADER
    );
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags2::SHADER_STORAGE_WRITE
    );
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags2::SHADER_READ);
    assert_eq!(barrier.old_layout, vk::ImageLayout::GENERAL);
    assert_eq!(barrier.new_layout, vk::ImageLayout::GENERAL);
}

#[test]
fn image_depth_attachment_read_write_split() {
    // Depth/stencil attachment read+write, with stencil read-only aspect.
    let image_barrier = make_image_barrier(
        &[vk_sync::AccessType::DepthStencilAttachmentWrite],
        &[vk_sync::AccessType::DepthAttachmentWriteStencilReadOnly],
        vk_sync::ImageLayout::Optimal,
        vk_sync::ImageLayout::Optimal,
        false,
    );

    let barrier = vk_sync::get_image_memory_barrier(&image_barrier);

    assert_eq!(
        barrier.src_stage_mask,
        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
    );
    assert_eq!(
        barrier.dst_stage_mask,
        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
    );
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
    );
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
    );
    assert_eq!(
        barrier.old_layout,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    );
    assert_eq!(
        barrier.new_layout,
        vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL
    );
}

#[test]
fn image_color_attachment_read_write_blending() {
    // Color attachment used with blending — both read and write.
    let image_barrier = make_image_barrier(
        &[vk_sync::AccessType::ColorAttachmentWrite],
        &[vk_sync::AccessType::ColorAttachmentReadWrite],
        vk_sync::ImageLayout::Optimal,
        vk_sync::ImageLayout::Optimal,
        false,
    );

    let barrier = vk_sync::get_image_memory_barrier(&image_barrier);

    assert_eq!(
        barrier.src_stage_mask,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
    );
    assert_eq!(
        barrier.dst_stage_mask,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
    );
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
    );
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
    );
    assert_eq!(
        barrier.old_layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );
    assert_eq!(
        barrier.new_layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );
}

#[test]
fn image_fragment_shading_rate_attachment() {
    // Producing a fragment shading rate attachment via compute, then sampling it.
    let image_barrier = make_image_barrier(
        &[vk_sync::AccessType::ComputeShaderWrite],
        &[vk_sync::AccessType::FragmentShadingRateAttachmentRead],
        vk_sync::ImageLayout::Optimal,
        vk_sync::ImageLayout::Optimal,
        false,
    );

    let barrier = vk_sync::get_image_memory_barrier(&image_barrier);

    assert_eq!(
        barrier.src_stage_mask,
        vk::PipelineStageFlags2::COMPUTE_SHADER
    );
    assert_eq!(
        barrier.dst_stage_mask,
        vk::PipelineStageFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR
    );
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags2::SHADER_STORAGE_WRITE
    );
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR
    );
    assert_eq!(barrier.old_layout, vk::ImageLayout::GENERAL);
    assert_eq!(
        barrier.new_layout,
        vk::ImageLayout::FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR
    );
}

#[test]
fn image_video_decode_to_fragment_sampled() {
    // Video decode output sampled by a fragment shader.
    let image_barrier = make_image_barrier(
        &[vk_sync::AccessType::VideoDecodeWrite],
        &[vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer],
        vk_sync::ImageLayout::Optimal,
        vk_sync::ImageLayout::Optimal,
        false,
    );

    let barrier = vk_sync::get_image_memory_barrier(&image_barrier);

    assert_eq!(
        barrier.src_stage_mask,
        vk::PipelineStageFlags2::VIDEO_DECODE_KHR
    );
    assert_eq!(
        barrier.dst_stage_mask,
        vk::PipelineStageFlags2::FRAGMENT_SHADER
    );
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR
    );
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags2::SHADER_SAMPLED_READ
    );
    assert_eq!(barrier.old_layout, vk::ImageLayout::VIDEO_DECODE_DST_KHR);
    assert_eq!(
        barrier.new_layout,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    );
}

#[test]
fn image_compute_write_optical_flow_input() {
    // Compute generates an image consumed by optical flow.
    let image_barrier = make_image_barrier(
        &[vk_sync::AccessType::ComputeShaderWrite],
        &[vk_sync::AccessType::OpticalFlowRead],
        vk_sync::ImageLayout::Optimal,
        vk_sync::ImageLayout::Optimal,
        false,
    );

    let barrier = vk_sync::get_image_memory_barrier(&image_barrier);

    assert_eq!(
        barrier.src_stage_mask,
        vk::PipelineStageFlags2::COMPUTE_SHADER
    );
    assert_eq!(
        barrier.dst_stage_mask,
        vk::PipelineStageFlags2::OPTICAL_FLOW_NV
    );
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags2::SHADER_STORAGE_WRITE
    );
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags2::OPTICAL_FLOW_READ_NV
    );
    assert_eq!(barrier.old_layout, vk::ImageLayout::GENERAL);
    assert_eq!(barrier.new_layout, vk::ImageLayout::GENERAL);
}

#[test]
fn image_queue_ownership_transfer() {
    // Release a color attachment from queue 0 and acquire it on queue 1.
    let image_barrier = vk_sync::ImageBarrier {
        previous_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
        next_accesses: &[vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer],
        previous_layout: vk_sync::ImageLayout::Optimal,
        next_layout: vk_sync::ImageLayout::Optimal,
        discard_contents: false,
        src_queue_family_index: 0,
        dst_queue_family_index: 1,
        image: vk::Image::null(),
        range: default_range(),
    };

    let barrier = vk_sync::get_image_memory_barrier(&image_barrier);

    assert_eq!(barrier.src_queue_family_index, 0);
    assert_eq!(barrier.dst_queue_family_index, 1);
    assert_eq!(
        barrier.src_stage_mask,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
    );
    assert_eq!(
        barrier.dst_stage_mask,
        vk::PipelineStageFlags2::FRAGMENT_SHADER
    );
    assert_eq!(
        barrier.old_layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );
    assert_eq!(
        barrier.new_layout,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    );
}
