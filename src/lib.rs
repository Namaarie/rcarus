mod camera;
mod hdr;
mod model;
mod resources;
mod texture;
mod util;
mod app;
mod state;
use cgmath::prelude::*;
use model::{DrawLight, DrawModel, Vertex};
use std::{iter, sync::Arc, time::Instant};
use texture::Texture;
use wgpu::util::DeviceExt;

// winit makes windows
use winit::{
    application::ApplicationHandler, dpi::PhysicalPosition, event::*, event_loop::{ActiveEventLoop, ControlFlow, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::{CursorGrabMode, Window, WindowId}
};

pub fn run() {
    // env_logger for wgpu errors
    env_logger::init();

    // winit event loop
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = app::App::default();
    let _ = event_loop.run_app(&mut app);
}
