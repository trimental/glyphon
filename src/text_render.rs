use crate::{
    custom_glyph::CustomGlyphCacheKey, ColorMode, ContentType, FontSystem, GlyphDetails,
    GlyphToRender, GpuCacheStatus, PrepareError, RasterizeCustomGlyphRequest,
    RasterizedCustomGlyph, RenderError, State, SwashCache, TextArea, TextAtlas,
    Viewport,
};
use cosmic_text::{Color, SubpixelBin, SwashContent};
use std::slice;
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, DepthStencilState, Device, Extent3d, MultisampleState,
    Origin3d, Queue, RenderPass, RenderPipeline, TexelCopyBufferLayout, TexelCopyTextureInfo,
    TextureAspect, COPY_BUFFER_ALIGNMENT,
};

/// A text renderer that uses cached glyphs to render text into an existing render pass.
pub struct TextRenderer {
    vertex_buffer: Buffer,
    vertex_buffer_size: u64,
    index_buffer: Buffer,
    index_buffer_size: u64,
    pipeline: RenderPipeline,
    glyph_vertices: Vec<GlyphToRender>,
    glyph_indices: Vec<u32>,
}

impl TextRenderer {
    /// Creates a new `TextRenderer`.
    pub fn new(
        atlas: &mut TextAtlas,
        device: &Device,
        multisample: MultisampleState,
        depth_stencil: Option<DepthStencilState>,
    ) -> Self {
        let vertex_buffer_size = next_copy_buffer_size(4096);
        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("glyphon vertices"),
            size: vertex_buffer_size,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer_size = next_copy_buffer_size(4096);
        let index_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("glyphon indices"),
            size: index_buffer_size,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = atlas.get_or_create_pipeline(device, multisample, depth_stencil);

        Self {
            vertex_buffer,
            vertex_buffer_size,
            index_buffer,
            index_buffer_size,
            pipeline,
            glyph_vertices: Vec::new(),
            glyph_indices: Vec::new(),
        }
    }

    /// Prepares all of the provided text areas for rendering.
    pub fn prepare<'a>(
        &mut self,
        device: &Device,
        queue: &Queue,
        font_system: &mut FontSystem,
        atlas: &mut TextAtlas,
        viewport: &Viewport,
        text_areas: impl IntoIterator<Item = TextArea<'a>>,
        cache: &mut SwashCache,
    ) -> Result<(), PrepareError> {
        self.prepare_with_depth_and_custom(
            device,
            queue,
            font_system,
            atlas,
            viewport,
            text_areas,
            cache,
            zero_depth,
            |_| None,
        )
    }

    /// Prepares all of the provided text areas for rendering.
    pub fn prepare_with_depth<'a>(
        &mut self,
        device: &Device,
        queue: &Queue,
        font_system: &mut FontSystem,
        atlas: &mut TextAtlas,
        viewport: &Viewport,
        text_areas: impl IntoIterator<Item = TextArea<'a>>,
        cache: &mut SwashCache,
        metadata_to_depth: impl FnMut(usize) -> f32,
    ) -> Result<(), PrepareError> {
        self.prepare_with_depth_and_custom(
            device,
            queue,
            font_system,
            atlas,
            viewport,
            text_areas,
            cache,
            metadata_to_depth,
            |_| None,
        )
    }

    /// Prepares all of the provided text areas for rendering.
    pub fn prepare_with_custom<'a>(
        &mut self,
        device: &Device,
        queue: &Queue,
        font_system: &mut FontSystem,
        atlas: &mut TextAtlas,
        viewport: &Viewport,
        text_areas: impl IntoIterator<Item = TextArea<'a>>,
        cache: &mut SwashCache,
        rasterize_custom_glyph: impl FnMut(RasterizeCustomGlyphRequest) -> Option<RasterizedCustomGlyph>,
    ) -> Result<(), PrepareError> {
        self.prepare_with_depth_and_custom(
            device,
            queue,
            font_system,
            atlas,
            viewport,
            text_areas,
            cache,
            zero_depth,
            rasterize_custom_glyph,
        )
    }

    /// Prepares all of the provided text areas for rendering.
    pub fn prepare_with_depth_and_custom<'a>(
        &mut self,
        device: &Device,
        queue: &Queue,
        font_system: &mut FontSystem,
        atlas: &mut TextAtlas,
        viewport: &Viewport,
        text_areas: impl IntoIterator<Item = TextArea<'a>>,
        cache: &mut SwashCache,
        mut metadata_to_depth: impl FnMut(usize) -> f32,
        mut rasterize_custom_glyph: impl FnMut(
            RasterizeCustomGlyphRequest,
        ) -> Option<RasterizedCustomGlyph>,
    ) -> Result<(), PrepareError> {
        self.glyph_vertices.clear();
        self.glyph_indices.clear();

        let state = State { device, queue };
        let mut system = GlyphSystem {
            atlas,
            cache,
            font_system,
        };
        let resolution = viewport.resolution();
        let view_proj = glam::Mat4::from_cols_array_2d(&viewport.params.view_proj);
        let res_w = resolution.width as f32;
        let res_h = resolution.height as f32;

        for text_area in text_areas {
            let is_identity =
                text_area.transform == glam::Mat4::IDENTITY && text_area.zoom == 1.0;

            let bounds = if is_identity {
                GlyphBounds {
                    x: Bounds {
                        min: text_area.bounds.left.max(0),
                        max: text_area.bounds.right.min(resolution.width as i32),
                    },
                    y: Bounds {
                        min: text_area.bounds.top.max(0),
                        max: text_area.bounds.bottom.min(resolution.height as i32),
                    },
                }
            } else {
                // Skip CPU clipping for transformed text; GPU handles it
                GlyphBounds {
                    x: Bounds {
                        min: i32::MIN / 2,
                        max: i32::MAX / 2,
                    },
                    y: Bounds {
                        min: i32::MIN / 2,
                        max: i32::MAX / 2,
                    },
                }
            };

            for glyph in text_area.custom_glyphs.iter() {
                let x = text_area.left + (glyph.left * text_area.scale);
                let y = text_area.top + (glyph.top * text_area.scale);
                let width = (glyph.width * text_area.scale).round() as u16;
                let height = (glyph.height * text_area.scale).round() as u16;

                let (x, y, x_bin, y_bin) = if glyph.snap_to_physical_pixel {
                    (
                        x.round() as i32,
                        y.round() as i32,
                        SubpixelBin::Zero,
                        SubpixelBin::Zero,
                    )
                } else {
                    let (x, x_bin) = SubpixelBin::new(x);
                    let (y, y_bin) = SubpixelBin::new(y);
                    (x, y, x_bin, y_bin)
                };

                let cache_key = GlyphonCacheKey::Custom(CustomGlyphCacheKey {
                    glyph_id: glyph.id,
                    width,
                    height,
                    x_bin,
                    y_bin,
                });

                let color = glyph.color.unwrap_or(text_area.default_color);

                if let Some(glyph_to_render) = prepare_glyph(
                    &state,
                    &mut system,
                    GlyphMetadata {
                        x,
                        y,
                        line_y: 0.0,
                        scale_factor: text_area.scale,
                        color,
                        metadata: glyph.metadata,
                        cache_key,
                    },
                    bounds,
                    |_system, rasterize_custom_glyph| -> Option<GetGlyphImageResult> {
                        if width == 0 || height == 0 {
                            return None;
                        }

                        let input = RasterizeCustomGlyphRequest {
                            id: glyph.id,
                            width,
                            height,
                            x_bin,
                            y_bin,
                            scale: text_area.scale,
                        };

                        let output = (rasterize_custom_glyph)(input)?;

                        output.validate(&input, None);

                        Some(GetGlyphImageResult {
                            content_type: output.content_type,
                            top: 0,
                            left: 0,
                            width,
                            height,
                            data: output.data,
                        })
                    },
                    &mut metadata_to_depth,
                    &mut rasterize_custom_glyph,
                )? {
                    expand_glyph(
                        glyph_to_render,
                        is_identity,
                        &text_area,
                        &view_proj,
                        res_w,
                        res_h,
                        &mut self.glyph_vertices,
                        &mut self.glyph_indices,
                    );
                }
            }

            let is_run_visible = |run: &cosmic_text::LayoutRun| {
                if !is_identity {
                    return true;
                }
                let start_y_physical = (text_area.top + (run.line_top * text_area.scale)) as i32;
                let end_y_physical = start_y_physical + (run.line_height * text_area.scale) as i32;

                start_y_physical <= text_area.bounds.bottom
                    && text_area.bounds.top <= end_y_physical
            };

            let layout_runs = text_area
                .buffer
                .layout_runs()
                .skip_while(|run| !is_run_visible(run))
                .take_while(is_run_visible);

            for run in layout_runs {
                for glyph in run.glyphs.iter() {
                    let physical_glyph =
                        glyph.physical((text_area.left, text_area.top), text_area.scale);

                    let color = match glyph.color_opt {
                        Some(some) => some,
                        None => text_area.default_color,
                    };

                    if let Some(glyph_to_render) = prepare_glyph(
                        &state,
                        &mut system,
                        GlyphMetadata {
                            x: physical_glyph.x,
                            y: physical_glyph.y,
                            line_y: run.line_y,
                            color,
                            metadata: glyph.metadata,
                            cache_key: GlyphonCacheKey::Text(physical_glyph.cache_key),
                            scale_factor: text_area.scale,
                        },
                        bounds,
                        |system, _rasterize_custom_glyph| -> Option<GetGlyphImageResult> {
                            let image = system
                                .cache
                                .get_image_uncached(system.font_system, physical_glyph.cache_key)?;

                            let (content_type, data) = match image.content {
                                SwashContent::Color => (ContentType::Color, image.data),
                                SwashContent::Mask => (ContentType::Mask, image.data),
                                SwashContent::SubpixelMask => {
                                    let width = image.placement.width as usize;
                                    let height = image.placement.height as usize;
                                    let pixel_count = width.saturating_mul(height);
                                    let mut data = image.data;
                                    if pixel_count > 0
                                        && data.len() >= pixel_count
                                        && data.len().is_multiple_of(pixel_count)
                                    {
                                        let channels = data.len() / pixel_count;
                                        if channels > 1 {
                                            let mut alpha = Vec::with_capacity(pixel_count);
                                            for px in data.chunks_exact(channels) {
                                                // Convert subpixel coverage to alpha coverage.
                                                alpha.push(*px.iter().max().unwrap_or(&0));
                                            }
                                            data = alpha;
                                        }
                                    }
                                    (ContentType::Mask, data)
                                }
                            };

                            Some(GetGlyphImageResult {
                                content_type,
                                top: image.placement.top as i16,
                                left: image.placement.left as i16,
                                width: image.placement.width as u16,
                                height: image.placement.height as u16,
                                data,
                            })
                        },
                        &mut metadata_to_depth,
                        &mut rasterize_custom_glyph,
                    )? {
                        expand_glyph(
                            glyph_to_render,
                            is_identity,
                            &text_area,
                            &view_proj,
                            res_w,
                            res_h,
                            &mut self.glyph_vertices,
                            &mut self.glyph_indices,
                        );
                    }
                }
            }
        }

        let will_render = !self.glyph_indices.is_empty();
        if !will_render {
            return Ok(());
        }

        // Upload vertex buffer
        let vertices = self.glyph_vertices.as_slice();
        let vertices_raw = unsafe {
            slice::from_raw_parts(
                vertices as *const _ as *const u8,
                std::mem::size_of_val(vertices),
            )
        };

        if self.vertex_buffer_size >= vertices_raw.len() as u64 {
            queue.write_buffer(&self.vertex_buffer, 0, vertices_raw);
        } else {
            self.vertex_buffer.destroy();

            let (buffer, buffer_size) = create_oversized_buffer(
                device,
                Some("glyphon vertices"),
                vertices_raw,
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            );

            self.vertex_buffer = buffer;
            self.vertex_buffer_size = buffer_size;
        }

        // Upload index buffer
        let indices = self.glyph_indices.as_slice();
        let indices_raw = unsafe {
            slice::from_raw_parts(
                indices as *const _ as *const u8,
                std::mem::size_of_val(indices),
            )
        };

        if self.index_buffer_size >= indices_raw.len() as u64 {
            queue.write_buffer(&self.index_buffer, 0, indices_raw);
        } else {
            self.index_buffer.destroy();

            let (buffer, buffer_size) = create_oversized_buffer(
                device,
                Some("glyphon indices"),
                indices_raw,
                BufferUsages::INDEX | BufferUsages::COPY_DST,
            );

            self.index_buffer = buffer;
            self.index_buffer_size = buffer_size;
        }

        Ok(())
    }

    /// Renders all layouts that were previously provided to `prepare`.
    pub fn render(
        &self,
        atlas: &TextAtlas,
        viewport: &Viewport,
        pass: &mut RenderPass<'_>,
    ) -> Result<(), RenderError> {
        if self.glyph_indices.is_empty() {
            return Ok(());
        }

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &atlas.bind_group, &[]);
        pass.set_bind_group(1, &viewport.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.glyph_indices.len() as u32, 0, 0..1);

        Ok(())
    }
}

#[repr(u16)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum TextColorConversion {
    None = 0,
    ConvertToLinear = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum GlyphonCacheKey {
    Text(cosmic_text::CacheKey),
    Custom(CustomGlyphCacheKey),
}

fn next_copy_buffer_size(size: u64) -> u64 {
    let align_mask = COPY_BUFFER_ALIGNMENT - 1;
    ((size.next_power_of_two() + align_mask) & !align_mask).max(COPY_BUFFER_ALIGNMENT)
}

fn create_oversized_buffer(
    device: &Device,
    label: Option<&str>,
    contents: &[u8],
    usage: BufferUsages,
) -> (Buffer, u64) {
    let size = next_copy_buffer_size(contents.len() as u64);
    let buffer = device.create_buffer(&BufferDescriptor {
        label,
        size,
        usage,
        mapped_at_creation: true,
    });
    buffer
        .slice(..)
        .get_mapped_range_mut()
        .slice(..contents.len())
        .copy_from_slice(contents);
    buffer.unmap();
    (buffer, size)
}

fn zero_depth(_: usize) -> f32 {
    0f32
}

/// Expands a single base glyph (from `prepare_glyph`) into 4 corner vertices and 6 indices.
fn expand_glyph(
    glyph: GlyphToRender,
    is_identity: bool,
    text_area: &TextArea,
    view_proj: &glam::Mat4,
    res_w: f32,
    res_h: f32,
    vertices: &mut Vec<GlyphToRender>,
    indices: &mut Vec<u32>,
) {
    let base_i = vertices.len() as u32;

    let gx = glyph.pos[0];
    let gy = glyph.pos[1];
    let gw = glyph.dim[0] as f32;
    let gh = glyph.dim[1] as f32;

    // 4 corner pixel positions: TL, TR, BR, BL
    let corners_px = [
        (gx, gy),
        (gx + gw, gy),
        (gx + gw, gy + gh),
        (gx, gy + gh),
    ];

    // 4 corner atlas UVs (pixel coordinates into the atlas)
    // Sample texel centers (handled in shader via +0.5 offset) to avoid
    // boundary artifacts from edge sampling at glyph atlas borders.
    let uv_right = glyph.uv[0].saturating_add(glyph.dim[0].saturating_sub(1));
    let uv_bottom = glyph.uv[1].saturating_add(glyph.dim[1].saturating_sub(1));
    let corners_uv: [[u16; 2]; 4] = [
        [glyph.uv[0], glyph.uv[1]],
        [uv_right, glyph.uv[1]],
        [uv_right, uv_bottom],
        [glyph.uv[0], uv_bottom],
    ];

    for i in 0..4 {
        let (px, py) = corners_px[i];

        let clip = if !is_identity {
            // Transform relative to text_area origin
            let local_x = px - text_area.left;
            let local_y = py - text_area.top;

            // Apply zoom then transform to world space
            let world_pos = text_area.transform
                * glam::Vec4::new(
                    local_x * text_area.zoom,
                    local_y * text_area.zoom,
                    glyph.depth,
                    1.0,
                );

            // view_proj maps directly from world space to clip space
            *view_proj * world_pos
        } else {
            // Screen-space text: convert pixel coords to NDC
            let ndc_x = 2.0 * px / res_w - 1.0;
            let ndc_y = 1.0 - 2.0 * py / res_h;
            *view_proj * glam::Vec4::new(ndc_x, ndc_y, glyph.depth, 1.0)
        };

        vertices.push(GlyphToRender {
            pos: [clip.x, clip.y, clip.w],
            dim: glyph.dim,
            uv: corners_uv[i],
            color: glyph.color,
            content_type_with_srgb: glyph.content_type_with_srgb,
            depth: clip.z,
        });
    }

    // Two triangles: (TL, TR, BR) and (TL, BR, BL)
    indices.extend_from_slice(&[
        base_i,
        base_i + 1,
        base_i + 2,
        base_i,
        base_i + 2,
        base_i + 3,
    ]);
}

struct GetGlyphImageResult {
    content_type: ContentType,
    top: i16,
    left: i16,
    width: u16,
    height: u16,
    data: Vec<u8>,
}

struct GlyphMetadata {
    x: i32,
    y: i32,
    line_y: f32,
    scale_factor: f32,
    color: Color,
    metadata: usize,
    cache_key: GlyphonCacheKey,
}

#[derive(Clone, Copy)]
struct Bounds {
    min: i32,
    max: i32,
}

#[derive(Clone, Copy)]
struct GlyphBounds {
    x: Bounds,
    y: Bounds,
}

struct GlyphSystem<'a> {
    atlas: &'a mut TextAtlas,
    cache: &'a mut SwashCache,
    font_system: &'a mut FontSystem,
}

fn prepare_glyph<R>(
    state: &State,
    system: &mut GlyphSystem,
    metadata: GlyphMetadata,
    bounds: GlyphBounds,
    get_glyph_image: impl FnOnce(&mut GlyphSystem, &mut R) -> Option<GetGlyphImageResult>,
    mut metadata_to_depth: impl FnMut(usize) -> f32,
    mut rasterize_custom_glyph: R,
) -> Result<Option<GlyphToRender>, PrepareError>
where
    R: FnMut(RasterizeCustomGlyphRequest) -> Option<RasterizedCustomGlyph>,
{
    let details =
        if let Some(details) = system.atlas.mask_atlas.glyph_cache.get(&metadata.cache_key) {
            system
                .atlas
                .mask_atlas
                .glyphs_in_use
                .insert(metadata.cache_key);
            details
        } else if let Some(details) = system
            .atlas
            .color_atlas
            .glyph_cache
            .get(&metadata.cache_key)
        {
            system
                .atlas
                .color_atlas
                .glyphs_in_use
                .insert(metadata.cache_key);
            details
        } else {
            let Some(image) = (get_glyph_image)(system, &mut rasterize_custom_glyph) else {
                return Ok(None);
            };

            let should_rasterize = image.width > 0 && image.height > 0;

            let (gpu_cache, atlas_id, inner) = if should_rasterize {
                let mut inner = system.atlas.inner_for_content_mut(image.content_type);

                // Find a position in the packer
                let allocation = loop {
                    match inner.try_allocate(image.width as usize, image.height as usize) {
                        Some(a) => break a,
                        None => {
                            if !system.atlas.grow(
                                state,
                                system.font_system,
                                system.cache,
                                image.content_type,
                                metadata.scale_factor,
                                &mut rasterize_custom_glyph,
                            ) {
                                return Err(PrepareError::AtlasFull);
                            }

                            inner = system.atlas.inner_for_content_mut(image.content_type);
                        }
                    }
                };
                let atlas_min = allocation.rectangle.min;

                state.queue.write_texture(
                    TexelCopyTextureInfo {
                        texture: &inner.texture,
                        mip_level: 0,
                        origin: Origin3d {
                            x: atlas_min.x as u32,
                            y: atlas_min.y as u32,
                            z: 0,
                        },
                        aspect: TextureAspect::All,
                    },
                    &image.data,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(image.width as u32 * inner.num_channels() as u32),
                        rows_per_image: None,
                    },
                    Extent3d {
                        width: image.width as u32,
                        height: image.height as u32,
                        depth_or_array_layers: 1,
                    },
                );

                (
                    GpuCacheStatus::InAtlas {
                        x: atlas_min.x as u16,
                        y: atlas_min.y as u16,
                        content_type: image.content_type,
                    },
                    Some(allocation.id),
                    inner,
                )
            } else {
                let inner = &mut system.atlas.color_atlas;
                (GpuCacheStatus::SkipRasterization, None, inner)
            };

            inner.glyphs_in_use.insert(metadata.cache_key);
            // Insert the glyph into the cache and return the details reference
            inner
                .glyph_cache
                .get_or_insert(metadata.cache_key, || GlyphDetails {
                    width: image.width,
                    height: image.height,
                    gpu_cache,
                    atlas_id,
                    top: image.top,
                    left: image.left,
                })
        };

    let mut x = metadata.x + details.left as i32;
    let mut y =
        (metadata.line_y * metadata.scale_factor).round() as i32 + metadata.y - details.top as i32;

    let (mut atlas_x, mut atlas_y, content_type) = match details.gpu_cache {
        GpuCacheStatus::InAtlas { x, y, content_type } => (x, y, content_type),
        GpuCacheStatus::SkipRasterization => return Ok(None),
    };

    let mut width = details.width as i32;
    let mut height = details.height as i32;

    // Starts beyond right edge or ends beyond left edge
    let max_x = x + width;
    if x > bounds.x.max || max_x < bounds.x.min {
        return Ok(None);
    }

    // Starts beyond bottom edge or ends beyond top edge
    let max_y = y + height;
    if y > bounds.y.max || max_y < bounds.y.min {
        return Ok(None);
    }

    // Clip left ege
    if x < bounds.x.min {
        let right_shift = bounds.x.min - x;

        x = bounds.x.min;
        width = max_x - bounds.x.min;
        atlas_x += right_shift as u16;
    }

    // Clip right edge
    if x + width > bounds.x.max {
        width = bounds.x.max - x;
    }

    // Clip top edge
    if y < bounds.y.min {
        let bottom_shift = bounds.y.min - y;

        y = bounds.y.min;
        height = max_y - bounds.y.min;
        atlas_y += bottom_shift as u16;
    }

    // Clip bottom edge
    if y + height > bounds.y.max {
        height = bounds.y.max - y;
    }

    let depth = metadata_to_depth(metadata.metadata);

    Ok(Some(GlyphToRender {
        pos: [x as f32, y as f32, 0.0],
        dim: [width as u16, height as u16],
        uv: [atlas_x, atlas_y],
        color: metadata.color.0,
        content_type_with_srgb: [
            content_type as u16,
            match system.atlas.color_mode {
                ColorMode::Accurate => TextColorConversion::ConvertToLinear,
                ColorMode::Web => TextColorConversion::None,
            } as u16,
        ],
        depth,
    }))
}
