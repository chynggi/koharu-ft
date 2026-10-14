//! Browser WebGPU canvas for prepared Koharu frames.
//!
//! Functional code is intentionally WASM-only. Native rendering, preparation,
//! persistence, and durable edit validation belong to the application and
//! `koharu-rasterizer`.

#[cfg(any(target_arch = "wasm32", test))]
mod cache;
// The ramp preview arithmetic is target-independent and compiled for the host
// under `cfg(test)` so `cargo test -p koharu-canvas --lib` can pin it, exactly
// as `cache` is.
#[cfg(any(target_arch = "wasm32", test))]
mod ramp;
#[cfg(target_arch = "wasm32")]
mod surface;
#[cfg(target_arch = "wasm32")]
mod web;

#[cfg(target_arch = "wasm32")]
pub use web::{WebCanvas, create_canvas};
