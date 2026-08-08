use super::*;
use ash;
use smallvec::SmallVec;

/// Number of barriers of each kind kept inline before spilling to the heap.
/// Sized for the common case of a handful of resources transitioning per
/// dependency; `vk::ImageMemoryBarrier2` is ~96 bytes, so this is ~768 bytes of
/// stack for the image array.
const INLINE_BARRIERS: usize = 8;

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
    let vk_memory_barrier = global_barrier.as_ref().map(get_memory_barrier);
    let vk_buffer_barriers: SmallVec<[vk::BufferMemoryBarrier2; INLINE_BARRIERS]> = buffer_barriers
        .iter()
        .map(get_buffer_memory_barrier)
        .collect();
    let vk_image_barriers: SmallVec<[vk::ImageMemoryBarrier2; INLINE_BARRIERS]> = image_barriers
        .iter()
        .map(get_image_memory_barrier)
        .collect();

    let dependency_info = vk::DependencyInfo::default()
        .memory_barriers(vk_memory_barrier.as_slice())
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
    let vk_memory_barrier = global_barrier.as_ref().map(get_memory_barrier);
    let vk_buffer_barriers: SmallVec<[vk::BufferMemoryBarrier2; INLINE_BARRIERS]> = buffer_barriers
        .iter()
        .map(get_buffer_memory_barrier)
        .collect();
    let vk_image_barriers: SmallVec<[vk::ImageMemoryBarrier2; INLINE_BARRIERS]> = image_barriers
        .iter()
        .map(get_image_memory_barrier)
        .collect();

    // `vkCmdWaitEvents2` requires one `VkDependencyInfo` per event, so duplicate
    // the same barrier set for every event.
    let dependency_infos: SmallVec<[vk::DependencyInfo; INLINE_BARRIERS]> = (0..events.len())
        .map(|_| {
            vk::DependencyInfo::default()
                .memory_barriers(vk_memory_barrier.as_slice())
                .buffer_memory_barriers(&vk_buffer_barriers)
                .image_memory_barriers(&vk_image_barriers)
        })
        .collect();

    unsafe {
        device.cmd_wait_events2(command_buffer, events, &dependency_infos);
    }
}
