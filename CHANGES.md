# Changes

## Unreleased

### Image layout resolution

Accesses within `previous_accesses` / `next_accesses` that require different image layouts are now
reconciled deterministically instead of tripping a `debug_assert` (which meant release builds
silently kept the first layout and emitted an incorrect barrier):

* Read-only layouts combine into `READ_ONLY_OPTIMAL`, attachment layouts into `ATTACHMENT_OPTIMAL`,
  and anything else into `GENERAL`. The resolution is order-independent, so `old_layout` and
  `new_layout` agree across successive barriers over the same access set.
* Accesses that admit no common layout (anything mixed with `Present`, a fragment density map or a
  video picture resource) are now reported rather than ignored.
* Added `try_get_image_memory_barrier`, returning `Result<_, LayoutError>`. `get_image_memory_barrier`
  is unchanged apart from panicking, rather than silently misbehaving, in those cases.
* Added an optional `log` feature that reports barriers whose layout had to be widened - legal, but
  potentially slower than intended.

**Behaviour changes to be aware of:**

* `AccessType::Nothing` (and any access implying no image layout, such as the buffer-only access
  types) is now ignored when resolving layouts rather than forcing `UNDEFINED`. Mixing it into
  `previous_accesses` alongside a real access therefore no longer discards the image contents - use
  `discard_contents` for that.
* `previous_accesses` must name the same access set that the barrier which last transitioned the
  subresource listed in its `next_accesses`, otherwise a widened `old_layout` will not match the
  image's actual layout. See the `ImageBarrier` docs.

### Fixes

* `new_layout` could be emitted as `UNDEFINED` - which Vulkan forbids - when `next_accesses` was
  empty, `[Nothing]`, or contained only buffer access types. Such barriers now leave the image in
  the layout it is already in, or report `LayoutError::UndefinedNewLayout` if that is unknown too.
* `ImageLayout::GeneralAndPresentation` was `unimplemented!()`; it now maps to `SHARED_PRESENT_KHR`.

### Performance

* `pipeline_barrier` and `wait_events` no longer heap-allocate for the common case: the global
  barrier uses `Option::as_slice` and the buffer/image barrier arrays use `SmallVec` with 8 inline
  elements.

## 0.5.0 (2025-01-25)

* Update to `ash` `0.38.0`.

## 0.4.0 (2022-04-06)

* Update to `ash` `0.37.0`.

## 0.3.0 (2021-12-28)

* Update to `ash` `0.35.0`.

## 0.2.2  (2021-12-10)

* Adds destination access marks to image barriers in all cases to avoid syncronisation hazards.

## 0.2.1 (2021-10-21)

* Changes the crate description to make it clearer in the crates browser that this crate is a fork.

## 0.2.0 (2021-10-21)

* Forked off of the original crate.
* Updated to ash 0.33.
* Added several `AccessType`s for the `VK_KHR_ray_tracing_pipeline` and `VK_KHR_acceleration_structure` extensions.
## 0.1.6 (2019-07-14)

* Removed inefficient Vec<AccessType> on barrier structs in favor of slice references.

## 0.1.5

* Updated to ash 0.29.

## 0.1.4

* Minor optimizations.

## 0.1.3

* Rust 2018 Edition.

## 0.1.2 (2018-11-17)

* Updated to ash 0.26
* Use default struct init from ash
* Made function pointer structs borrowed for performance
* Some minor cleanup

## 0.1.1 (2018-11-15)

* Updated to ash 0.25 (Vulkan 1.1)
* Added support for NVX generated commands
* Added support for read-only depth/stencil + writeable depth/stencil
* Added Copy and Default traits to AccessType and ImageLayout
* Added Debug, Default, and Clone traits to GlobalBarrier, BufferBarrier, and ImageBarrier

## 0.1.0 (2018-08-26)

* First release
