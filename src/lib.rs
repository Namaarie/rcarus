mod app;
mod camera;
mod hdr;
mod model;
mod resources;
mod texture;
mod util;

use winit::event_loop::{ControlFlow, EventLoop};

pub fn run() {
    // env_logger for wgpu errors
    // env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = app::App::new();
    let _ = event_loop.run_app(&mut app);
}
