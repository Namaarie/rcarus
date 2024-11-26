use glam::{Mat4, Vec3, Vec4};
use std::f32::consts::FRAC_PI_2;
use winit::dpi::PhysicalPosition;
use winit::event::*;
use winit::keyboard::KeyCode;

const SAFE_FRAC_PI_2: f32 = FRAC_PI_2 - 0.0001;

// required for rust to store data correctly for shaders
#[repr(C)]
// so we can store into buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_position: Vec4,
    view: Mat4,
    view_proj: Mat4,
    inv_proj: Mat4,
    inv_view: Mat4,
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_position: Vec4::ZERO,
            view: Mat4::IDENTITY,
            view_proj: Mat4::IDENTITY,
            inv_proj: Mat4::IDENTITY,
            inv_view: Mat4::IDENTITY,
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera, projection: &Projection) {
        self.view_position =
            Vec4::from_array([camera.position.x, camera.position.y, camera.position.z, 1.0]);
        let proj = projection.calc_matrix();
        let view = camera.calc_matrix();
        let view_proj = proj * view;
        self.view = view;
        self.view_proj = view_proj;
        self.inv_proj = proj.inverse();
        self.inv_view = view.transpose();
    }
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Mat4 = Mat4::from_cols(
    Vec4::from_array([1.0, 0.0, 0.0, 0.0]),
    Vec4::from_array([0.0, 1.0, 0.0, 0.0]),
    Vec4::from_array([0.0, 0.0, 0.5, 0.0]),
    Vec4::from_array([0.0, 0.0, 0.5, 1.0])
);

pub struct Projection {
    aspect: f32,
    fovy_rad: f32,
    znear: f32,
    zfar: f32,
}

pub struct Camera {
    pub position: Vec3,
    yaw_rad: f32,
    pitch_rad: f32,
}

pub struct CameraController {
    amount_left: f32,
    amount_right: f32,
    amount_forward: f32,
    amount_backward: f32,
    amount_up: f32,
    amount_down: f32,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    scroll: f32,
    speed: f32,
    sensitivity: f32,
}

impl Camera {
    pub fn new(position: Vec3, yaw: f32, pitch: f32) -> Self {
        Self {
            position: position,
            yaw_rad: yaw,
            pitch_rad: pitch,
        }
    }

    pub fn calc_matrix(&self) -> Mat4 {
        let (sin_pitch, cos_pitch) = self.pitch_rad.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw_rad.sin_cos();

        Mat4::look_to_rh(
            self.position,
            Vec3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize(),
            Vec3::from_array([0.0, 1.0, 0.0]),
        )
    }
}

impl Projection {
    pub fn new(width: u32, height: u32, fovy: f32, znear: f32, zfar: f32) -> Self {
        Self {
            aspect: width as f32 / height as f32,
            fovy_rad: fovy,
            znear,
            zfar,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    pub fn calc_matrix(&self) -> Mat4 {
        OPENGL_TO_WGPU_MATRIX * glam::Mat4::perspective_rh(self.fovy_rad, self.aspect, self.znear, self.zfar)
    }
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            amount_left: 0.0,
            amount_right: 0.0,
            amount_forward: 0.0,
            amount_backward: 0.0,
            amount_up: 0.0,
            amount_down: 0.0,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            scroll: 0.0,
            speed,
            sensitivity,
        }
    }

    pub fn process_keyboard(&mut self, key: KeyCode, state: ElementState) -> bool {
        let amount = if state == ElementState::Pressed {
            1.0
        } else {
            0.0
        };
        match key {
            KeyCode::KeyW => {
                self.amount_forward = amount;
                true
            }
            KeyCode::KeyS => {
                self.amount_backward = amount;
                true
            }
            KeyCode::KeyA => {
                self.amount_left = amount;
                true
            }
            KeyCode::KeyD => {
                self.amount_right = amount;
                true
            }
            KeyCode::Space => {
                self.amount_up = amount;
                true
            }
            KeyCode::ShiftLeft => {
                self.amount_down = amount;
                true
            }
            _ => false,
        }
    }

    pub fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
        self.rotate_horizontal = mouse_dx as f32;
        self.rotate_vertical = mouse_dy as f32;
    }

    pub fn process_scroll(&mut self, delta: &MouseScrollDelta) {
        self.scroll = match delta {
            // I'm assuming a line is about 100 pixels
            MouseScrollDelta::LineDelta(_, scroll) => scroll * 100.0,
            MouseScrollDelta::PixelDelta(PhysicalPosition { y: scroll, .. }) => *scroll as f32,
        };
    }

    pub fn update_camera_position(&mut self, camera: &mut Camera, dt: f32) {
        //println!("{}", self.current_time.elapsed().as_secs_f64());
        // Move forward/backward and left/right
        let (yaw_sin, yaw_cos) = camera.yaw_rad.sin_cos();
        let forward = Vec3::new(yaw_cos, 0.0, yaw_sin).normalize();
        let right = Vec3::new(-yaw_sin, 0.0, yaw_cos).normalize();

        let mut pos = camera.position;
        pos += forward * (self.amount_forward - self.amount_backward);
        pos += right * (self.amount_right - self.amount_left);

        // Move in/out (aka. "zoom")
        // Note: this isn't an actual zoom. The camera's position
        // changes when zooming. I've added this to make it easier
        // to get closer to an object you want to focus on.
        let (pitch_sin, pitch_cos) = camera.pitch_rad.sin_cos();
        let scrollward =
            Vec3::new(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin).normalize();
        camera.position += scrollward * self.scroll * self.speed * self.sensitivity;

        self.scroll = 0.0;

        // Move up/down. Since we don't use roll, we can just
        // modify the y coordinate directly.
        pos.y += self.amount_up - self.amount_down;

        if pos.distance(camera.position) > 0.0 {
            camera.position += (pos - camera.position).normalize() * self.speed * dt;
        }
    }

    pub fn update_camera_rotation(&mut self, camera: &mut Camera) {
        // Rotate
        camera.yaw_rad += self.rotate_horizontal * self.sensitivity;
        camera.pitch_rad += -self.rotate_vertical * self.sensitivity;

        // println!("{:?} {:?}", camera.yaw, camera.pitch);

        // If process_mouse isn't called every frame, these values
        // will not get set to zero, and the camera will rotate
        // when moving in a non-cardinal direction.
        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        // Keep the camera's angle from going too high/low.
        if camera.pitch_rad < -SAFE_FRAC_PI_2 {
            camera.pitch_rad = -SAFE_FRAC_PI_2;
        } else if camera.pitch_rad > SAFE_FRAC_PI_2 {
            camera.pitch_rad = SAFE_FRAC_PI_2;
        }
    }
}
