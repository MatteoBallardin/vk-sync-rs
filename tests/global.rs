//! Tests are based on the common synchronization examples on the Vulkan-Docs wiki: https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples.

use ash::vk;
use vk_sync_fork as vk_sync;

#[test]
fn compute_write_storage_compute_read_storage() {
    // Compute write to storage buffer/image, Compute read from storage buffer/image
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
        next_accesses: &[vk_sync::AccessType::ComputeShaderReadOther],
    };

    let barrier = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::COMPUTE_SHADER);
    assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::COMPUTE_SHADER);
    assert_eq!(barrier.src_access_mask, vk::AccessFlags2::SHADER_STORAGE_WRITE);
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags2::SHADER_READ);
}

#[test]
fn compute_read_storage_compute_write_storage() {
    // Compute read from storage buffer, Compute write from storage buffer
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
        next_accesses: &[vk_sync::AccessType::ComputeShaderReadOther],
    };

    let barrier = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::COMPUTE_SHADER);
    assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::COMPUTE_SHADER);
    assert_eq!(barrier.src_access_mask, vk::AccessFlags2::SHADER_STORAGE_WRITE);
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags2::SHADER_READ);
}

#[test]
fn compute_write_storage_graphics_read_index() {
    // Compute write to storage buffer, Graphics read as index buffer
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
        next_accesses: &[vk_sync::AccessType::IndexBuffer],
    };

    let barrier = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::COMPUTE_SHADER);
    assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::INDEX_INPUT);
    assert_eq!(barrier.src_access_mask, vk::AccessFlags2::SHADER_STORAGE_WRITE);
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags2::INDEX_READ);
}

#[test]
fn compute_write_storage_graphics_read_indirect() {
    // Compute write to storage buffer, Graphics read as indirect buffer
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
        next_accesses: &[vk_sync::AccessType::IndirectBuffer],
    };

    let barrier = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::COMPUTE_SHADER);
    assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::DRAW_INDIRECT);
    assert_eq!(barrier.src_access_mask, vk::AccessFlags2::SHADER_STORAGE_WRITE);
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags2::INDIRECT_COMMAND_READ
    );
}

#[test]
fn nothing_transfer_read() {
    // None, Transfer read from buffer
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::Nothing],
        next_accesses: &[vk_sync::AccessType::TransferRead],
    };

    let barrier = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::NONE);
    assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::ALL_TRANSFER);
    assert_eq!(barrier.src_access_mask, vk::AccessFlags2::empty());
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags2::empty());
}

#[test]
fn transfer_write_graphics_read_vertex() {
    // Transfer write to buffer, Graphics read from vertex buffer
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::TransferWrite],
        next_accesses: &[vk_sync::AccessType::VertexBuffer],
    };

    let barrier = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::ALL_TRANSFER);
    assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT);
    assert_eq!(barrier.src_access_mask, vk::AccessFlags2::TRANSFER_WRITE);
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags2::VERTEX_ATTRIBUTE_READ
    );
}

#[test]
fn full_pipeline_barrier() {
    // Full pipeline barrier
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::General],
        next_accesses: &[vk_sync::AccessType::General],
    };

    let barrier = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::ALL_COMMANDS);
    assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::ALL_COMMANDS);
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE
    );
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE
    );
}

#[test]
fn compute_write_storage_graphics_read_index_compute_read_uniform() {
    // Compute write to storage buffer, Graphics read as index buffer & Compute read as uniform buffer
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
        next_accesses: &[
            vk_sync::AccessType::IndexBuffer,
            vk_sync::AccessType::ComputeShaderReadUniformBuffer,
        ],
    };

    let barrier = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::COMPUTE_SHADER);
    assert_eq!(
        barrier.dst_stage_mask,
        vk::PipelineStageFlags2::INDEX_INPUT | vk::PipelineStageFlags2::COMPUTE_SHADER
    );
    assert_eq!(barrier.src_access_mask, vk::AccessFlags2::SHADER_STORAGE_WRITE);
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags2::INDEX_READ | vk::AccessFlags2::UNIFORM_READ
    );
}

#[test]
fn compute_write_texel_graphics_read_indirect_fragment_read_uniform() {
    // Compute write to storage texel buffer, Graphics read as indirect buffer & fragment read as uniform buffer
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
        next_accesses: &[
            vk_sync::AccessType::IndirectBuffer,
            vk_sync::AccessType::FragmentShaderReadUniformBuffer,
        ],
    };

    let barrier = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::COMPUTE_SHADER);
    assert_eq!(
        barrier.dst_stage_mask,
        vk::PipelineStageFlags2::DRAW_INDIRECT | vk::PipelineStageFlags2::FRAGMENT_SHADER
    );
    assert_eq!(barrier.src_access_mask, vk::AccessFlags2::SHADER_STORAGE_WRITE);
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags2::INDIRECT_COMMAND_READ | vk::AccessFlags2::UNIFORM_READ
    );
}
