//! Tests for image layout resolution when a barrier's access list requires more than one layout
//! at once - widening, unresolvable conflicts, and the `UNDEFINED` edge cases.

use ash::vk;
use vk_sync_fork as vk_sync;

use vk_sync::{AccessType, BarrierDirection, ImageLayout, LayoutError};

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
    previous: &'a [AccessType],
    next: &'a [AccessType],
    previous_layout: ImageLayout,
    next_layout: ImageLayout,
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

/// Resolve an `Optimal`/`Optimal` barrier, panicking on error.
fn optimal<'a>(previous: &'a [AccessType], next: &'a [AccessType]) -> vk::ImageMemoryBarrier2<'a> {
    let barrier = make_image_barrier(
        previous,
        next,
        ImageLayout::Optimal,
        ImageLayout::Optimal,
        false,
    );
    vk_sync::get_image_memory_barrier(&barrier)
}

// ---------------------------------------------------------------------------
// Widening
// ---------------------------------------------------------------------------

#[test]
fn read_only_accesses_widen_to_read_only_optimal() {
    // Sampled read and depth/stencil attachment read want SHADER_READ_ONLY_OPTIMAL and
    // DEPTH_STENCIL_READ_ONLY_OPTIMAL respectively; READ_ONLY_OPTIMAL serves both.
    let barrier = optimal(
        &[AccessType::DepthStencilAttachmentWrite],
        &[
            AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
            AccessType::DepthStencilAttachmentRead,
        ],
    );

    assert_eq!(barrier.new_layout, vk::ImageLayout::READ_ONLY_OPTIMAL);
}

#[test]
fn attachment_accesses_widen_to_attachment_optimal() {
    // Full depth/stencil write plus a depth-write/stencil-read aspect split.
    let barrier = optimal(
        &[AccessType::Nothing],
        &[
            AccessType::DepthStencilAttachmentWrite,
            AccessType::DepthAttachmentWriteStencilReadOnly,
        ],
    );

    assert_eq!(barrier.new_layout, vk::ImageLayout::ATTACHMENT_OPTIMAL);
}

#[test]
fn unrelated_accesses_widen_to_general() {
    // TRANSFER_SRC_OPTIMAL is read-only, but READ_ONLY_OPTIMAL does not permit transfer reads,
    // so this must fall all the way back to GENERAL rather than to READ_ONLY_OPTIMAL.
    let barrier = optimal(
        &[AccessType::ColorAttachmentWrite],
        &[
            AccessType::TransferRead,
            AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
        ],
    );

    assert_eq!(barrier.new_layout, vk::ImageLayout::GENERAL);
}

#[test]
fn widening_is_order_independent() {
    // Every permutation of a conflicting set must resolve identically, otherwise old_layout and
    // new_layout could disagree across successive barriers over the same set.
    let permutations = [
        [
            AccessType::TransferRead,
            AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
            AccessType::ColorAttachmentWrite,
        ],
        [
            AccessType::TransferRead,
            AccessType::ColorAttachmentWrite,
            AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
        ],
        [
            AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
            AccessType::TransferRead,
            AccessType::ColorAttachmentWrite,
        ],
        [
            AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
            AccessType::ColorAttachmentWrite,
            AccessType::TransferRead,
        ],
        [
            AccessType::ColorAttachmentWrite,
            AccessType::TransferRead,
            AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
        ],
        [
            AccessType::ColorAttachmentWrite,
            AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
            AccessType::TransferRead,
        ],
    ];

    for permutation in &permutations {
        let barrier = optimal(&[AccessType::Nothing], permutation);
        assert_eq!(
            barrier.new_layout,
            vk::ImageLayout::GENERAL,
            "order dependence for {permutation:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Unresolvable conflicts
// ---------------------------------------------------------------------------

#[test]
fn present_cannot_be_reconciled_with_other_accesses() {
    let barrier = make_image_barrier(
        &[AccessType::Nothing],
        &[AccessType::Present, AccessType::ColorAttachmentWrite],
        ImageLayout::Optimal,
        ImageLayout::Optimal,
        false,
    );

    match vk_sync::try_get_image_memory_barrier(&barrier) {
        Err(LayoutError::Conflict {
            direction,
            resolved,
            access,
            required,
        }) => {
            assert_eq!(direction, BarrierDirection::Next);
            assert_eq!(resolved, vk::ImageLayout::PRESENT_SRC_KHR);
            assert_eq!(access, AccessType::ColorAttachmentWrite);
            assert_eq!(required, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        }
        other => panic!("expected a layout conflict, got {other:?}"),
    }
}

#[test]
#[should_panic(expected = "conflicting image layouts in next_accesses")]
fn infallible_mapping_panics_on_conflict() {
    let _ = optimal(
        &[AccessType::Nothing],
        &[AccessType::Present, AccessType::ColorAttachmentWrite],
    );
}

#[test]
fn conflicts_are_reported_for_previous_accesses_too() {
    let barrier = make_image_barrier(
        &[AccessType::Present, AccessType::TransferWrite],
        &[AccessType::ColorAttachmentWrite],
        ImageLayout::Optimal,
        ImageLayout::Optimal,
        false,
    );

    match vk_sync::try_get_image_memory_barrier(&barrier) {
        Err(LayoutError::Conflict { direction, .. }) => {
            assert_eq!(direction, BarrierDirection::Previous);
        }
        other => panic!("expected a layout conflict, got {other:?}"),
    }
}

#[test]
fn general_layout_still_conflicts_with_present() {
    // Under ImageLayout::General, Present maps to PRESENT_SRC_KHR while everything else maps to
    // GENERAL - so mixing them has no resolution.
    let barrier = make_image_barrier(
        &[AccessType::Nothing],
        &[AccessType::Present, AccessType::ComputeShaderWrite],
        ImageLayout::Optimal,
        ImageLayout::General,
        false,
    );

    assert!(matches!(
        vk_sync::try_get_image_memory_barrier(&barrier),
        Err(LayoutError::Conflict { .. })
    ));
}

// ---------------------------------------------------------------------------
// UNDEFINED handling
// ---------------------------------------------------------------------------

#[test]
fn nothing_is_neutral_and_does_not_discard() {
    // `Nothing` implies no layout, so it must not drag the resolved layout back to UNDEFINED and
    // silently discard the image contents.
    let barrier = optimal(
        &[AccessType::Nothing, AccessType::ColorAttachmentWrite],
        &[AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer],
    );

    assert_eq!(
        barrier.old_layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );

    // ... and the same holds whichever order they were listed in.
    let reversed = optimal(
        &[AccessType::ColorAttachmentWrite, AccessType::Nothing],
        &[AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer],
    );

    assert_eq!(reversed.old_layout, barrier.old_layout);
}

#[test]
fn new_layout_is_never_undefined() {
    // Buffer-only access types imply no image layout. Rather than emitting the illegal
    // newLayout == UNDEFINED, the barrier leaves the image in the layout it is already in.
    let barrier = optimal(
        &[AccessType::ColorAttachmentWrite],
        &[AccessType::IndexBuffer],
    );

    assert_eq!(
        barrier.new_layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );
    assert_eq!(barrier.new_layout, barrier.old_layout);
}

#[test]
fn empty_next_accesses_reuses_the_previous_layout() {
    let barrier = optimal(&[AccessType::ColorAttachmentWrite], &[]);

    assert_eq!(
        barrier.new_layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );
}

#[test]
fn barrier_with_no_layout_at_all_is_an_error() {
    let barrier = make_image_barrier(&[], &[], ImageLayout::Optimal, ImageLayout::Optimal, false);

    assert!(matches!(
        vk_sync::try_get_image_memory_barrier(&barrier),
        Err(LayoutError::UndefinedNewLayout)
    ));
}

#[test]
fn discard_still_forces_an_undefined_old_layout() {
    let barrier = make_image_barrier(
        &[AccessType::Present, AccessType::Present],
        &[AccessType::ColorAttachmentWrite],
        ImageLayout::Optimal,
        ImageLayout::Optimal,
        true,
    );
    let barrier = vk_sync::get_image_memory_barrier(&barrier);

    assert_eq!(barrier.old_layout, vk::ImageLayout::UNDEFINED);
    assert_eq!(
        barrier.new_layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );
}

#[test]
fn discarded_barrier_falls_back_to_the_pre_discard_layout() {
    // old_layout is discarded to UNDEFINED, but the fallback for an absent new layout must use
    // the layout resolved *before* the discard, not the clobbered one.
    let barrier = make_image_barrier(
        &[AccessType::ColorAttachmentWrite],
        &[AccessType::IndexBuffer],
        ImageLayout::Optimal,
        ImageLayout::Optimal,
        true,
    );
    let barrier = vk_sync::get_image_memory_barrier(&barrier);

    assert_eq!(barrier.old_layout, vk::ImageLayout::UNDEFINED);
    assert_eq!(
        barrier.new_layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );
}

// ---------------------------------------------------------------------------
// Shared presentable images
// ---------------------------------------------------------------------------

#[test]
fn general_and_presentation_maps_to_shared_present() {
    // Previously `unimplemented!()`. Every access uses SHARED_PRESENT_KHR, including Present, so
    // no combination can conflict.
    let barrier = make_image_barrier(
        &[AccessType::ColorAttachmentWrite, AccessType::Present],
        &[AccessType::ComputeShaderWrite, AccessType::Present],
        ImageLayout::GeneralAndPresentation,
        ImageLayout::GeneralAndPresentation,
        false,
    );
    let barrier = vk_sync::get_image_memory_barrier(&barrier);

    assert_eq!(barrier.old_layout, vk::ImageLayout::SHARED_PRESENT_KHR);
    assert_eq!(barrier.new_layout, vk::ImageLayout::SHARED_PRESENT_KHR);
}
