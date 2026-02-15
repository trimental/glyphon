use crate::{Cache, Params, Resolution};
use std::{mem, slice};
use wgpu::{BindGroup, Buffer, BufferDescriptor, BufferUsages, Device, Queue};

/// A camera uniform containing the view-projection matrix.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CameraUniform {
    /// The view-projection matrix. Use `glam::Mat4::IDENTITY.to_cols_array_2d()` for no
    /// camera transformation.
    pub view_proj: [[f32; 4]; 4],
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
        }
    }
}

/// Controls the visible area of all text for a given renderer. Any text outside of the visible
/// area will be clipped.
///
/// Many projects will only ever need a single `Viewport`, but it is possible to create multiple
/// `Viewport`s if you want to render text to specific areas within a window (without having to)
/// bound each `TextArea`).
#[derive(Debug)]
pub struct Viewport {
    pub(crate) params: Params,
    params_buffer: Buffer,
    pub(crate) bind_group: BindGroup,
}

impl Viewport {
    /// Creates a new `Viewport` with the given `device` and `cache`.
    pub fn new(device: &Device, cache: &Cache) -> Self {
        let params = Params {
            screen_resolution: Resolution {
                width: 0,
                height: 0,
            },
            _pad: [0, 0],
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
        };

        let params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("glyphon params"),
            size: mem::size_of::<Params>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = cache.create_uniforms_bind_group(device, &params_buffer);

        Self {
            params,
            params_buffer,
            bind_group,
        }
    }

    /// Updates the `Viewport` with the given `resolution` and `camera`.
    pub fn update(&mut self, queue: &Queue, resolution: Resolution, camera: CameraUniform) {
        self.params.screen_resolution = resolution;
        self.params.view_proj = camera.view_proj;

        queue.write_buffer(&self.params_buffer, 0, unsafe {
            slice::from_raw_parts(
                &self.params as *const Params as *const u8,
                mem::size_of::<Params>(),
            )
        });
    }

    /// Returns the current resolution of the `Viewport`.
    pub fn resolution(&self) -> Resolution {
        self.params.screen_resolution
    }
}
