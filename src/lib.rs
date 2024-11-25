mod camera;
mod hdr;
mod model;
mod resources;
mod texture;
mod util;
use cgmath::prelude::*;
use model::{DrawLight, DrawModel, Vertex};
use std::{iter, time::Instant};
use texture::Texture;
use wgpu::util::DeviceExt;

// winit makes windows
use winit::{
    dpi::PhysicalPosition,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowBuilder},
};

struct Config {
    vsync_mode: wgpu::PresentMode,
    camera_speed: f32,
    camera_sensitivity: f32,
    backend: wgpu::Backends,
}

const CONFIG: Config = Config {
    vsync_mode: wgpu::PresentMode::AutoVsync,
    backend: wgpu::Backends::DX12,
    camera_speed: 5.0,
    camera_sensitivity: 0.002,
};

const NUM_INSTANCES_HEIGHT: u32 = 1;
const NUM_INSTANCES_PER_ROW: u32 = 1000;
const SPACE_BETWEEN: f32 = 1.0;

#[derive(Default)]
struct MouseGrabber {
    manual_lock: bool,
}

impl MouseGrabber {
    // Call this with true to lock and false to unlock
    fn grab(&mut self, window: &Window, grab: bool) {
        if grab != self.manual_lock {
            if grab {
                if window.set_cursor_grab(CursorGrabMode::Locked).is_err() {
                    window.set_cursor_grab(CursorGrabMode::Confined).unwrap();
                    let size = window.inner_size();
                    let _ = window.set_cursor_position(PhysicalPosition {
                        x: size.width / 2,
                        y: size.height / 2,
                    });
                    self.manual_lock = true;
                }
            } else {
                self.manual_lock = false;
                window.set_cursor_grab(CursorGrabMode::None).unwrap();
            }
            window.set_cursor_visible(!grab);
            // println!("manual lock is {:?}", self.manual_lock);
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform {
    position: [f32; 3],
    // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
    _padding: u32,
    color: [f32; 3],
    // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
    _padding2: u32,
}

struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        let model =
            cgmath::Matrix4::from_translation(self.position) * cgmath::Matrix4::from(self.rotation);
        InstanceRaw {
            model: model.into(),
            normal: cgmath::Matrix3::from(self.rotation).into(),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    normal: [[f32; 3]; 3],
}

impl model::Vertex for InstanceRaw {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
                // for each vec4. We'll have to reassemble the mat4 in the shader.
                wgpu::VertexAttribute {
                    offset: 0,
                    // While our vertex shader only uses locations 0, and 1 now, in later tutorials, we'll
                    // be using 2, 3, and 4, for Vertex. We'll start at slot 5, not conflict with them later
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 19]>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 22]>() as wgpu::BufferAddress,
                    shader_location: 12,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GlobalsUniform {
    time: f32,
}

impl GlobalsUniform {
    fn new() -> Self {
        Self { time: 0.0 }
    }

    fn update_time(&mut self, delta_time: f32) {
        self.time += delta_time;
    }
}

// required for rust to store data correctly for shaders
#[repr(C)]
// so we can store into buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    // We can't use cgmath with bytemuck directly, so we'll have
    // to convert the Matrix4 into a 4x4 f32 array
    view_position: [f32; 4],
    view: [[f32; 4]; 4],
    view_proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    inv_view: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_position: [0.0; 4],
            view: cgmath::Matrix4::identity().into(),
            view_proj: cgmath::Matrix4::identity().into(),
            inv_proj: cgmath::Matrix4::identity().into(), // NEW!
            inv_view: cgmath::Matrix4::identity().into(), // NEW!
        }
    }

    fn update_view_proj(&mut self, camera: &camera::Camera, projection: &camera::Projection) {
        self.view_position = camera.position.to_homogeneous().into();
        let proj = projection.calc_matrix();
        let view = camera.calc_matrix();
        let view_proj = proj * view;
        self.view = view.into();
        self.view_proj = view_proj.into();
        self.inv_proj = proj.invert().unwrap().into();
        self.inv_view = view.transpose().into();
    }
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    topology: wgpu::PrimitiveTopology,
    shader: wgpu::ShaderModuleDescriptor,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(shader);

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("{:?}", shader)),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            // vertex buffer
            buffers: vertex_layouts,
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                // how to blend between old and new pixels
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                // writes to all colors rgba
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
            polygon_mode: wgpu::PolygonMode::Fill,
            // Requires Features::DEPTH_CLIP_CONTROL
            unclipped_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
            conservative: false,
        },
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
}

struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    // The window must be declared after the surface so
    // it gets dropped after it as the surface contains
    // unsafe references to the window's resources.
    window: &'a Window,
    clear_color: [f64; 3],
    render_pipeline: wgpu::RenderPipeline,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    // diffuse_bind_group: wgpu::BindGroup,
    // diffuse_texture: texture::Texture,
    camera: camera::Camera,
    projection: camera::Projection,
    camera_controller: camera::CameraController,
    camera_uniform: CameraUniform, // buffers holds the uniforms
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    globals_bind_group: wgpu::BindGroup,
    globals_uniform: GlobalsUniform, // buffers holds the uniforms
    globals_buffer: wgpu::Buffer,
    start_time: Instant,
    current_time: Instant,
    delta_time_f32: f32,
    delta_time_f64: f64,
    mouse_pressed: bool,
    mouse_grabber: MouseGrabber,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
    depth_texture: Texture,
    obj_model: model::Model,
    adapter: wgpu::Adapter,
    light_uniform: LightUniform,
    light_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    light_render_pipeline: wgpu::RenderPipeline,
    hdr: hdr::HdrPipeline,
    environment_bind_group: wgpu::BindGroup,
    sky_pipeline: wgpu::RenderPipeline,
}

impl<'a> State<'a> {
    // Creating some of the wgpu types requires async code
    async fn new(window: &'a Window) -> anyhow::Result<State<'a>> {
        let start_time = Instant::now();
        let delta_time_f32 = 0.0;
        let delta_time_f64 = 0.0;
        let current_time = Instant::now();

        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: CONFIG.backend,
            ..Default::default()
        });

        // create surface using GPU instance
        let surface = instance.create_surface(window).unwrap();

        // adapter is handle for actual GPU
        // request gpu with specified options
        let adapter = instance
            .request_adapter(
                // options for gpu adapter
                &wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                },
                // request_adapter is async, so wait for results then unwrap
            )
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    // debug label for device
                    label: None,
                    // memory allocation strategy
                    memory_hints: Default::default(),
                },
                // trace_path is api call tracing, if that feature is enabled in `wgpu-core`
                None,
            )
            .await
            .unwrap();

        // surface capabilities
        let surface_caps = surface.get_capabilities(&adapter);

        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result in all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        // finds surface format that is srgb or default to formats[0] which is preferred
        // Rgba8Unorm is final texture in srgb, shader in srgb
        // Rgba8UnormSrgb is final texture in srgb, shader in linear rgb
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        // surface config
        let config = wgpu::SurfaceConfiguration {
            // sets the surface texture as an output texture of a render pass
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            // srgb format
            format: surface_format,
            width: size.width,
            height: size.height,
            // vsync mode, FIFO is generic vysnc on
            present_mode: CONFIG.vsync_mode,
            // alpha blending, prob with transparent windows
            alpha_mode: surface_caps.alpha_modes[0],
            // ???
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let hdr = hdr::HdrPipeline::new(&device, &config);

        let hdr_loader = resources::HdrLoader::new(&device);
        let sky_bytes = resources::load_binary("pure-sky.hdr").await?;
        let sky_texture = hdr_loader.from_equirectangular_bytes(
            &device,
            &queue,
            &sky_bytes,
            1080,
            Some("Sky Texture"),
        )?;

        let environment_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("environment_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });

        let environment_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("environment_bind_group"),
            layout: &environment_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&sky_texture.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sky_texture.sampler()),
                },
            ],
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // normal map
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        // roll is rotation around forward axis (+-Z)
        // pitch is rotation around X axis basically up/down
        // yaw is rotation around Y axis basically left/right
        let camera = camera::Camera::new((0.0, 1.0, 5.0), cgmath::Deg(-90.0), cgmath::Deg(0.0));
        let projection =
            camera::Projection::new(config.width, config.height, cgmath::Deg(60.0), 0.1, 10000.0);
        let camera_controller =
            camera::CameraController::new(CONFIG.camera_speed, CONFIG.camera_sensitivity);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    // camera needs to be known at vertex shader
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let globals_uniform = GlobalsUniform::new();

        let globals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Globals Buffer"),
            contents: bytemuck::cast_slice(&[globals_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let globals_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("globals_bind_group_layout"),
            });

        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &globals_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            }],
            label: Some("globals_bind_group"),
        });

        let clear_color = [1.0, 0.9254901960784314, 0.8392156862745098];

        let light_uniform = LightUniform {
            position: [2.0, 2.0, 2.0],
            _padding: 0,
            color: [1.0, 1.0, 1.0],
            _padding2: 0,
        };

        // We'll want to update our lights position, so we use COPY_DST
        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light VB"),
            contents: bytemuck::cast_slice(&[light_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: None,
            });

        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
            label: None,
        });

        // render pipeline layout
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                // defines which bind group each buffer is in
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_bind_group_layout,
                    &light_bind_group_layout,
                    &environment_layout,
                ],
                push_constant_ranges: &[],
            });

        // render pipeline
        let render_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Normal Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            };

            create_render_pipeline(
                &device,
                &render_pipeline_layout,
                hdr.format(),
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), InstanceRaw::desc()],
                wgpu::PrimitiveTopology::TriangleList,
                shader,
            )
        };

        let light_render_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Light Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &light_bind_group_layout],
                push_constant_ranges: &[],
            });

            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Light Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("light.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &layout,
                hdr.format(),
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc()],
                wgpu::PrimitiveTopology::TriangleList,
                shader,
            )
        };

        let sky_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Sky Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &environment_layout],
                push_constant_ranges: &[],
            });
            let shader = wgpu::include_wgsl!("sky.wgsl");
            create_render_pipeline(
                &device,
                &layout,
                hdr.format(),
                Some(texture::Texture::DEPTH_FORMAT),
                &[],
                wgpu::PrimitiveTopology::TriangleList,
                shader,
            )
        };

        let mut mouse_grabber = MouseGrabber {
            ..Default::default()
        };

        mouse_grabber.grab(window, true);

        let instances = (0..NUM_INSTANCES_HEIGHT)
            .flat_map(|y| {
                (0..NUM_INSTANCES_PER_ROW).flat_map(move |z| {
                    (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                        let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                        let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                        let y = SPACE_BETWEEN * (y as f32);

                        //let y = ((x/100.0).sin() + (z/100.0).sin()).abs()*25.0;

                        let position = cgmath::Vector3 {
                            x: x as f32,
                            y: y as f32,
                            z: z as f32,
                        };

                        // let rotation = if position.is_zero() {
                        //     // this is needed so an object at (0, 0, 0) won't get scaled to zero
                        //     // as Quaternions can affect scale if they're not created correctly
                        //     cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
                        // } else {
                        //     cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
                        // };

                        // let rotation = cgmath::Quaternion::from_axis_angle((0.0, 1.0, 0.0).into(), cgmath::Deg(180.0));

                        let rotation = cgmath::Quaternion::one();
                        Instance { position, rotation }
                    })
                })
            })
            .collect::<Vec<_>>();

        // const SPACE_BETWEEN: f32 = 2.0;
        // let instances = (0..NUM_INSTANCES_PER_ROW)
        //     .flat_map(|z| {
        //         (0..NUM_INSTANCES_PER_ROW).map(move |x| {
        //             let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
        //             let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);

        //             let position = cgmath::Vector3 { x, y: 0.0, z };

        // 			let rotation = cgmath::Quaternion::one();

        //             Instance { position, rotation }
        //         })
        //     })
        //     .collect::<Vec<_>>();

        // let mut pos = cgmath::Vector2::new(25.0, 0.0);
        // let angle: f32 = std::f32::consts::FRAC_1_SQRT_2;
        // let instances = (0..1_000_000).map(|i| {
        //     let rot = (((i as f32) / 1_000_000_f32) * 1000.0);
        //     let theta = rot * 1000.0;
        //     let r = (rot * 100.0).sin()*50.0;

        //     let y = rot + current_time.elapsed().as_secs_f32().sin() * 100.0;

        //     let x = r * theta.cos() * y.sqrt() * 2.0 + r * 5.0;
        //     let z = r * theta.sin() * y.log2() * 2.0 + r * 5.0 + x.sin();

        //     let rotation = cgmath::Quaternion::one();
        //     let position = cgmath::Vector3{x, y, z};
        //     Instance {
        //         position, rotation
        //     }
        // }).collect::<Vec<_>>();

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        // let obj_model = resources::load_model("cube.obj", &device, &queue, &texture_bind_group_layout)
        //     .await
        //     .unwrap();

        let obj_model = resources::load_cube(&device, &queue, &texture_bind_group_layout)
            .await
            .unwrap();

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            clear_color,
            render_pipeline,
            // diffuse_bind_group,
            // diffuse_texture,
            texture_bind_group_layout,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_controller,
            globals_bind_group,
            globals_uniform,
            globals_buffer,
            start_time,
            delta_time_f32,
            delta_time_f64,
            projection,
            mouse_pressed: false,
            mouse_grabber,
            current_time,
            instances,
            instance_buffer,
            depth_texture,
            obj_model,
            adapter,
            light_uniform,
            light_buffer,
            light_bind_group,
            light_render_pipeline,
            hdr,
            environment_bind_group,
            sky_pipeline,
        })
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        // make sure new_size is greater than 0 since can't have 0 size window
        if new_size.width > 0 && new_size.height > 0 {
            // resize camera projection
            self.projection.resize(new_size.width, new_size.height);

            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            // resize depth texture
            self.depth_texture =
                texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");

            self.hdr
                .resize(&self.device, new_size.width, new_size.height);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        // println!("inputting");
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => match key {
                KeyCode::Escape => {
                    self.mouse_grabber.grab(&self.window, false);
                    true
                }
                _ => self.camera_controller.process_keyboard(*key, *state),
            },
            WindowEvent::MouseWheel { delta, .. } => {
                self.camera_controller.process_scroll(delta);
                true
            }
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                true
            }
            _ => false,
        }
    }

    fn update_delta_time(&mut self) {
        self.delta_time_f32 = self.current_time.elapsed().as_secs_f32();
        self.delta_time_f64 = self.current_time.elapsed().as_secs_f64();
        self.current_time = Instant::now();
    }

    fn update_camera_rotation(&mut self) {
        self.camera_controller
            .update_camera_rotation(&mut self.camera);
    }

    fn update(&mut self) {
        // println!("updating");
        // println!("{:?}", self.config.format);

        let fps = 1.0 / self.delta_time_f64;

        let info: wgpu::AdapterInfo = self.adapter.get_info();
        if let Some(report) = self.device.generate_allocator_report() {
            self.window.set_title(&format!(
                "VRAM: {} / {} | {} v{} on {} at {}fps",
                util::FmtBytes(report.total_allocated_bytes),
                util::FmtBytes(report.total_reserved_bytes),
                info.name,
                info.driver,
                info.backend,
                fps,
            ));
        } else {
            self.window.set_title(&format!(
                "VRAM: ? / ? | {} v{} on {} at {}fps",
                info.name, info.driver, info.backend, fps,
            ));
        }

        if self.mouse_pressed {
            self.mouse_grabber.grab(&self.window, true);
        }

        // println!("{:?}", self.camera.position);
        // println!("{}", delta_time);
        // println!("{:?}", self.mouse_pressed);
        // self.camera_controller.update_camera(&mut self.camera, self.delta_time_f32);
        //println!("{}", self.current_time.elapsed().as_secs_f64());

        self.camera_controller
            .update_camera_position(&mut self.camera, self.delta_time_f32);
        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        let old_position: cgmath::Vector3<_> = self.light_uniform.position.into();
        let t = self.start_time.elapsed().as_secs_f32();
        //self.light_uniform.position = (cgmath::Quaternion::from_axis_angle((0.0, 1.0, 0.0).into(), cgmath::Deg(std::f32::consts::PI * self.delta_time_f32 * 100.0)) * old_position).into();

        //self.light_uniform.position[1] = self.light_uniform.position[1] + t.sin() * 2.0;

        //self.light_uniform.color = [(t.sin()/2.0 + 0.5) as f32, (t.cos()/2.0 + 0.5) as f32, ((t+std::f64::consts::PI).sin()/2.0 + 0.5) as f32];

        self.queue.write_buffer(
            &self.light_buffer,
            0,
            bytemuck::cast_slice(&[self.light_uniform]),
        );

        self.light_uniform.position[1] = old_position[1];

        self.globals_uniform.update_time(self.delta_time_f32);
        self.queue.write_buffer(
            &self.globals_buffer,
            0,
            bytemuck::cast_slice(&[self.globals_uniform]),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.config.format.add_srgb_suffix()),
            ..Default::default()
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.hdr.view(), // UPDATED!
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_pipeline(&self.light_render_pipeline);
            render_pass.draw_light_model(
                &self.obj_model,
                &self.camera_bind_group,
                &self.light_bind_group,
            );

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.draw_model_instanced(
                &self.obj_model,
                0..self.instances.len() as u32,
                &self.camera_bind_group,
                &self.light_bind_group,
                &self.environment_bind_group,
            );

            render_pass.set_pipeline(&self.sky_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.environment_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        // NEW!
        // Apply tonemapping
        self.hdr.process(&mut encoder, &view);

        #[cfg(feature = "debug")]
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Debug"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            self.debug.draw_axis(&mut pass, &self.camera_bind_group);
        }

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub async fn run() {
    // env_logger for wgpu errors
    env_logger::init();

    // winit event loop
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    //println!("{:?}", event_loop.control_flow());

    // actual window
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    //window.set_title(&format!("wgpu-voxels {:?}", CONFIG.vsync_mode));

    // get state, need to await since async
    let mut state = State::new(&window).await.unwrap();

    // window event loop
    if let Err(event_loop_err) = event_loop.run(|event, control_flow| {
        // event is the event
        // control_flow allows us to control the event loop, to exit
        match event {
            // device events
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                if state.mouse_grabber.manual_lock {
                    state.camera_controller.process_mouse(delta.0, delta.1);
                    state.update_camera_rotation();
                }
            }

            // window events
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window.id() && !state.input(event) => {
                // if event isn't proceesed by state.input
                match event {
                    // exit if matched close requested
                    WindowEvent::CloseRequested => control_flow.exit(),

                    // window resize event
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }

                    // ignore other key presses/other events
                    _ => {}
                }
            }
            // ignore events that are not window events
            _ => {}
        }

        // then match for redraw
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window.id() => {
                match event {
                    WindowEvent::RedrawRequested => {
                        match state.render() {
                            Ok(_) => {
                                state.update_delta_time();
                                state.update();
                            }

                            // Reconfigure the surface if it's lost or outdated
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                state.resize(state.size)
                            }

                            // The system is out of memory, we should probably quit
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                log::error!("OutOfMemory");
                                control_flow.exit();
                            }

                            // This happens when the a frame takes too long to present
                            Err(wgpu::SurfaceError::Timeout) => {
                                log::warn!("Surface timeout")
                            }
                        }

                        // This tells winit that we want another frame after this one
                        state.window().request_redraw();
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {}
            _ => {}
        }
    }) {
        // event loop error
        println!("event loop error: {:?}", event_loop_err);
    }
}
