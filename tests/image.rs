//! Tests are based on the common synchronization examples on the Vulkan-Docs wiki: https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples.

use ash::vk;
use vk_sync_fork as vk_sync;

#[test]
fn compute_write_storage_fragment_read_sampled() {
    // Compute write to storage image, Graphics fragment read as sampled image
    let image_barrier = vk_sync::ImageBarrier {
        previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
        next_accesses: &[vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer],
        previous_layout: vk_sync::ImageLayout::Optimal,
        next_layout: vk_sync::ImageLayout::Optimal,
        discard_contents: false,
        src_queue_family_index: 0,
        dst_queue_family_index: 0,
        image: vk::Image::null(),
        range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::empty(),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
    };

    let barrier = vk_sync::get_image_memory_barrier(&image_barrier);

    assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::COMPUTE_SHADER);
    assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::FRAGMENT_SHADER);
    assert_eq!(barrier.src_access_mask, vk::AccessFlags2::SHADER_STORAGE_WRITE);
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags2::SHADER_SAMPLED_READ);
    assert_eq!(barrier.old_layout, vk::ImageLayout::GENERAL);
    assert_eq!(
        barrier.new_layout,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    );
}

#[test]
fn graphics_write_color_compute_read_sampled() {
    // Graphics write to color attachment, Compute read from sampled image
    let image_barrier = vk_sync::ImageBarrier {
        previous_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
        next_accesses: &[vk_sync::AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer],
        previous_layout: vk_sync::ImageLayout::Optimal,
        next_layout: vk_sync::ImageLayout::Optimal,
        discard_contents: false,
        src_queue_family_index: 0,
        dst_queue_family_index: 0,
        image: vk::Image::null(),
        range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::empty(),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
    };

    let barrier = vk_sync::get_image_memory_barrier(&image_barrier);

    assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
    assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::COMPUTE_SHADER);
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
    );
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags2::SHADER_SAMPLED_READ);
    assert_eq!(
        barrier.old_layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );
    assert_eq!(
        barrier.new_layout,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    );
}

#[test]
fn graphics_write_depth_compute_read_sampled() {
    // Graphics write to color attachment, Compute read from sampled image
    let image_barrier = vk_sync::ImageBarrier {
        previous_accesses: &[vk_sync::AccessType::DepthStencilAttachmentWrite],
        next_accesses: &[vk_sync::AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer],
        previous_layout: vk_sync::ImageLayout::Optimal,
        next_layout: vk_sync::ImageLayout::Optimal,
        discard_contents: false,
        src_queue_family_index: 0,
        dst_queue_family_index: 0,
        image: vk::Image::null(),
        range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::empty(),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
    };

    let barrier = vk_sync::get_image_memory_barrier(&image_barrier);

    assert_eq!(
        barrier.src_stage_mask,
        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
    );
    assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::COMPUTE_SHADER);
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
    );
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags2::SHADER_SAMPLED_READ);
    assert_eq!(
        barrier.old_layout,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    );
    assert_eq!(
        barrier.new_layout,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    );
}

#[test]
fn graphics_write_depth_fragment_read_attachment() {
    // Graphics write to depth attachment, Graphics fragment read from input attachment
    let image_barrier = vk_sync::ImageBarrier {
        previous_accesses: &[vk_sync::AccessType::DepthStencilAttachmentWrite],
        next_accesses: &[vk_sync::AccessType::FragmentShaderReadDepthStencilInputAttachment],
        previous_layout: vk_sync::ImageLayout::Optimal,
        next_layout: vk_sync::ImageLayout::Optimal,
        discard_contents: false,
        src_queue_family_index: 0,
        dst_queue_family_index: 0,
        image: vk::Image::null(),
        range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::empty(),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
    };

    let barrier = vk_sync::get_image_memory_barrier(&image_barrier);

    assert_eq!(
        barrier.src_stage_mask,
        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
    );
    assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::FRAGMENT_SHADER);
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
    );
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags2::INPUT_ATTACHMENT_READ
    );
    assert_eq!(
        barrier.old_layout,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    );
    assert_eq!(
        barrier.new_layout,
        vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
    );
}

#[test]
fn graphics_write_depth_fragment_read_sampled() {
    // Graphics write to depth attachment, Graphics fragment read from sampled image
    let image_barrier = vk_sync::ImageBarrier {
        previous_accesses: &[vk_sync::AccessType::DepthStencilAttachmentWrite],
        next_accesses: &[vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer],
        previous_layout: vk_sync::ImageLayout::Optimal,
        next_layout: vk_sync::ImageLayout::Optimal,
        discard_contents: false,
        src_queue_family_index: 0,
        dst_queue_family_index: 0,
        image: vk::Image::null(),
        range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::empty(),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
    };

    let barrier = vk_sync::get_image_memory_barrier(&image_barrier);

    assert_eq!(
        barrier.src_stage_mask,
        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
    );
    assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::FRAGMENT_SHADER);
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
    );
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags2::SHADER_SAMPLED_READ);
    assert_eq!(
        barrier.old_layout,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    );
    assert_eq!(
        barrier.new_layout,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    );
}

#[test]
fn graphics_write_color_fragment_read_attachment() {
    // Graphics write to color attachment, Graphics fragment read from input attachment
    let image_barrier = vk_sync::ImageBarrier {
        previous_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
        next_accesses: &[vk_sync::AccessType::FragmentShaderReadColorInputAttachment],
        previous_layout: vk_sync::ImageLayout::Optimal,
        next_layout: vk_sync::ImageLayout::Optimal,
        discard_contents: false,
        src_queue_family_index: 0,
        dst_queue_family_index: 0,
        image: vk::Image::null(),
        range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::empty(),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
    };

    let barrier = vk_sync::get_image_memory_barrier(&image_barrier);

    assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
    assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::FRAGMENT_SHADER);
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
    );
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags2::INPUT_ATTACHMENT_READ
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

#[test]
fn graphics_write_color_fragment_read_sampled() {
    // Graphics write to color attachment, Graphics fragment read from input attachment
    let image_barrier = vk_sync::ImageBarrier {
        previous_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
        next_accesses: &[vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer],
        previous_layout: vk_sync::ImageLayout::Optimal,
        next_layout: vk_sync::ImageLayout::Optimal,
        discard_contents: false,
        src_queue_family_index: 0,
        dst_queue_family_index: 0,
        image: vk::Image::null(),
        range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::empty(),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
    };

    let barrier = vk_sync::get_image_memory_barrier(&image_barrier);

    assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
    assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::FRAGMENT_SHADER);
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
    );
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags2::SHADER_SAMPLED_READ);
    assert_eq!(
        barrier.old_layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );
    assert_eq!(
        barrier.new_layout,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    );
}

#[test]
fn graphics_write_color_vertex_read_sampled() {
    // Graphics write to color attachment, Graphics vertex read from sampled image
    let image_barrier = vk_sync::ImageBarrier {
        previous_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
        next_accesses: &[vk_sync::AccessType::VertexShaderReadSampledImageOrUniformTexelBuffer],
        previous_layout: vk_sync::ImageLayout::Optimal,
        next_layout: vk_sync::ImageLayout::Optimal,
        discard_contents: false,
        src_queue_family_index: 0,
        dst_queue_family_index: 0,
        image: vk::Image::null(),
        range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::empty(),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
    };

    let barrier = vk_sync::get_image_memory_barrier(&image_barrier);

    assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
    assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::VERTEX_SHADER);
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
    );
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags2::SHADER_SAMPLED_READ);
    assert_eq!(
        barrier.old_layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );
    assert_eq!(
        barrier.new_layout,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    );
}

#[test]
fn graphics_read_sampled_graphics_write_color() {
    // Graphics fragment read from sampled image, Graphics write to color attachment
    let image_barrier = vk_sync::ImageBarrier {
        previous_accesses: &[
            vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
        ],
        next_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
        previous_layout: vk_sync::ImageLayout::Optimal,
        next_layout: vk_sync::ImageLayout::Optimal,
        discard_contents: false,
        src_queue_family_index: 0,
        dst_queue_family_index: 0,
        image: vk::Image::null(),
        range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::empty(),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
    };

    let barrier = vk_sync::get_image_memory_barrier(&image_barrier);

    assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::FRAGMENT_SHADER);
    assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
    assert_eq!(barrier.src_access_mask, vk::AccessFlags2::empty());
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
    );
    assert_eq!(
        barrier.old_layout,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    );
    assert_eq!(
        barrier.new_layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );
}

#[test]
fn transfer_write_image_fragment_read_sampled() {
    // Transfer write to image, Graphics fragment read from sampled image
    let image_barrier = vk_sync::ImageBarrier {
        previous_accesses: &[vk_sync::AccessType::TransferWrite],
        next_accesses: &[vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer],
        previous_layout: vk_sync::ImageLayout::Optimal,
        next_layout: vk_sync::ImageLayout::Optimal,
        discard_contents: false,
        src_queue_family_index: 0,
        dst_queue_family_index: 0,
        image: vk::Image::null(),
        range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::empty(),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
    };

    let barrier = vk_sync::get_image_memory_barrier(&image_barrier);

    assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::ALL_TRANSFER);
    assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::FRAGMENT_SHADER);
    assert_eq!(barrier.src_access_mask, vk::AccessFlags2::TRANSFER_WRITE);
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags2::SHADER_SAMPLED_READ);
    assert_eq!(barrier.old_layout, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
    assert_eq!(
        barrier.new_layout,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    );
}

#[test]
fn graphics_write_color_presentation() {
    // Graphics color attachment write, Presentation
    let image_barrier = vk_sync::ImageBarrier {
        previous_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
        next_accesses: &[vk_sync::AccessType::Present],
        previous_layout: vk_sync::ImageLayout::Optimal,
        next_layout: vk_sync::ImageLayout::Optimal,
        discard_contents: false,
        src_queue_family_index: 0,
        dst_queue_family_index: 0,
        image: vk::Image::null(),
        range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::empty(),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
    };

    let barrier = vk_sync::get_image_memory_barrier(&image_barrier);

    assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
    assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::NONE);
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
    );
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags2::empty());
    assert_eq!(
        barrier.old_layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );
    assert_eq!(barrier.new_layout, vk::ImageLayout::PRESENT_SRC_KHR);
}
