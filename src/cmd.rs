use super::*;
use ash;

/// Simplified wrapper around `vkCmdPipelineBarrier2`.
/// The mapping functions defined above are used to translate the passed in
/// barrier definitions into the synchronization 2 barrier structs that are
/// then bundled into a `vk::DependencyInfo` and submitted via
/// `vkCmdPipelineBarrier2`.
/// `command_buffer` is passed unmodified to `vkCmdPipelineBarrier2`.
pub fn pipeline_barrier(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    global_barrier: Option<GlobalBarrier>,
    buffer_barriers: &[BufferBarrier],
    image_barriers: &[ImageBarrier],
) {
    // TODO: Optimize out the Vec heap allocations
    let mut vk_memory_barriers: Vec<vk::MemoryBarrier2> = Vec::with_capacity(1);
    let mut vk_buffer_barriers: Vec<vk::BufferMemoryBarrier2> =
        Vec::with_capacity(buffer_barriers.len());
    let mut vk_image_barriers: Vec<vk::ImageMemoryBarrier2> =
        Vec::with_capacity(image_barriers.len());

    if let Some(ref barrier) = global_barrier {
        vk_memory_barriers.push(get_memory_barrier(barrier));
    }

    for buffer_barrier in buffer_barriers {
        vk_buffer_barriers.push(get_buffer_memory_barrier(buffer_barrier));
    }

    for image_barrier in image_barriers {
        vk_image_barriers.push(get_image_memory_barrier(image_barrier));
    }

    let dependency_info = vk::DependencyInfo::default()
        .memory_barriers(&vk_memory_barriers)
        .buffer_memory_barriers(&vk_buffer_barriers)
        .image_memory_barriers(&vk_image_barriers);

    unsafe {
        device.cmd_pipeline_barrier2(command_buffer, &dependency_info);
    }
}

/// Wrapper around `vkCmdSetEvent2`.
/// Sets an event when the accesses defined by `previous_accesses` are completed.
/// `command_buffer` and `event` are passed unmodified to `vkCmdSetEvent2`.
pub fn set_event(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    event: vk::Event,
    previous_accesses: &[AccessType],
) {
    let mut stage_mask = vk::PipelineStageFlags2::empty();
    for previous_access in previous_accesses {
        let previous_info = get_access_info(*previous_access);
        stage_mask |= previous_info.stage_mask;
    }

    let memory_barrier = vk::MemoryBarrier2::default().src_stage_mask(stage_mask);
    let memory_barriers = [memory_barrier];
    let dependency_info = vk::DependencyInfo::default().memory_barriers(&memory_barriers);

    unsafe {
        device.cmd_set_event2(command_buffer, event, &dependency_info);
    }
}

/// Wrapper around `vkCmdResetEvent2`.
/// Resets an event when the accesses defined by `previous_accesses` are completed.
/// `command_buffer` and `event` are passed unmodified to `vkCmdResetEvent2`.
pub fn reset_event(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    event: vk::Event,
    previous_accesses: &[AccessType],
) {
    let mut stage_mask = vk::PipelineStageFlags2::empty();
    for previous_access in previous_accesses {
        let previous_info = get_access_info(*previous_access);
        stage_mask |= previous_info.stage_mask;
    }

    unsafe {
        device.cmd_reset_event2(command_buffer, event, stage_mask);
    }
}

/// Simplified wrapper around `vkCmdWaitEvents2`.
/// The mapping functions defined above are used to translate the passed in
/// barrier definitions into the synchronization 2 barrier structs that are
/// then bundled into a `vk::DependencyInfo` per event and submitted via
/// `vkCmdWaitEvents2`.
///
/// `commandBuffer` and `events` are passed unmodified to `vkCmdWaitEvents2`.
pub fn wait_events(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    events: &[vk::Event],
    global_barrier: Option<GlobalBarrier>,
    buffer_barriers: &[BufferBarrier],
    image_barriers: &[ImageBarrier],
) {
    // TODO: Optimize out the Vec heap allocations
    let mut vk_memory_barriers: Vec<vk::MemoryBarrier2> = Vec::with_capacity(1);
    let mut vk_buffer_barriers: Vec<vk::BufferMemoryBarrier2> =
        Vec::with_capacity(buffer_barriers.len());
    let mut vk_image_barriers: Vec<vk::ImageMemoryBarrier2> =
        Vec::with_capacity(image_barriers.len());

    if let Some(ref barrier) = global_barrier {
        vk_memory_barriers.push(get_memory_barrier(barrier));
    }

    for buffer_barrier in buffer_barriers {
        vk_buffer_barriers.push(get_buffer_memory_barrier(buffer_barrier));
    }

    for image_barrier in image_barriers {
        vk_image_barriers.push(get_image_memory_barrier(image_barrier));
    }

    // `vkCmdWaitEvents2` requires one `VkDependencyInfo` per event, so duplicate
    // the same barrier set for every event.
    let dependency_infos: Vec<vk::DependencyInfo> = (0..events.len())
        .map(|_| {
            vk::DependencyInfo::default()
                .memory_barriers(&vk_memory_barriers)
                .buffer_memory_barriers(&vk_buffer_barriers)
                .image_memory_barriers(&vk_image_barriers)
        })
        .collect();

    unsafe {
        device.cmd_wait_events2(command_buffer, events, &dependency_infos);
    }
}
