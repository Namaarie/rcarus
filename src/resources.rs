use crate::{
    model::{self, Mesh, ModelVertex},
    texture,
};
use anyhow::{Context, Ok};
use image::codecs::hdr::HdrDecoder;
use std::{
    env,
    fmt::format,
    io::{BufReader, Cursor},
    os,
    path::PathBuf,
};
use wgpu::util::DeviceExt;

//     E--------F
//    /|       /|
//   / |      / |
//  A--|-----B  |
//  |  G-----|--H
//  | /      | /
//  |/       |/
//  C--------D
// centered around 0, 0, 0
// size of 1
// +Z is out from monitor, -Z is forward into monitor
// +X is right, +Y is up
// indices = ACD, DBA, BDH, HFB, FHG, GEF, EAB, BFE, CGH, HDC, EGC, CAE
// 0, 2, 3, 3, 1, 0,
// 1, 3, 7, 7, 5, 1,
// 5, 7, 6, 6, 4, 5,
// 4, 0, 1, 1, 5, 4,
// 2, 6, 7, 7, 3, 2,
// 4, 6, 2, 2, 0, 4
// 1 ---- 4
// |      |
// |      |
// 2 ---- 3
// pub const CUBE_VERTICES: &[ModelVertex] = &[
//     ModelVertex {
//         // A0
//         position: [-0.5, 0.5, 0.5],
//         tex_coords: [0.0, 0.0],
//         normal: [0.0, 0.0, 0.0],
//         color: [1.0, 1.0, 1.0],
//     },
//     ModelVertex {
//         // B1
//         position: [0.5, 0.5, 0.5],
//         tex_coords: [1.0, 0.0],
//         normal: [0.0, 0.0, 0.0],
//         color: [1.0, 1.0, 1.0],
//     },
//     ModelVertex {
//         // C2
//         position: [-0.5, -0.5, 0.5],
//         tex_coords: [0.0, 1.0],
//         normal: [0.0, 0.0, 0.0],
//         color: [1.0, 1.0, 1.0],
//     },
//     ModelVertex {
//         // D3
//         position: [0.5, -0.5, 0.5],
//         tex_coords: [1.0, 1.0],
//         normal: [0.0, 0.0, 0.0],
//         color: [1.0, 1.0, 1.0],
//     },
//     ModelVertex {
//         // E4
//         position: [-0.5, 0.5, -0.5],
//         tex_coords: [1.0, 0.0],
//         normal: [0.0, 0.0, 0.0],
//         color: [1.0, 1.0, 1.0],
//     },
//     ModelVertex {
//         // F5
//         position: [0.5, 0.5, -0.5],
//         tex_coords: [0.0, 0.0],
//         normal: [0.0, 0.0, 0.0],
//         color: [1.0, 1.0, 1.0],
//     },
//     ModelVertex {
//         // G6
//         position: [-0.5, -0.5, -0.5],
//         tex_coords: [1.0, 1.0],
//         normal: [0.0, 0.0, 0.0],
//         color: [1.0, 1.0, 1.0],
//     },
//     ModelVertex {
//         // H7
//         position: [0.5, -0.5, -0.5],
//         tex_coords: [0.0, 1.0],
//         normal: [0.0, 0.0, 0.0],
//         color: [1.0, 1.0, 1.0],
//     },
// ];

// pub const CUBE_INDICES: &[u32] = &[
//     0, 2, 3, 3, 1, 0, 1, 3, 7, 7, 5, 1, 5, 7, 6, 6, 4, 5, 4, 0, 1, 1, 5, 4, 2, 6, 7, 7, 3, 2, 4, 6, 2, 2, 0, 4
// ];

pub async fn load_texture(
    file_name: &str,
    is_normal_map: bool,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mag_filter_mode: wgpu::FilterMode,
) -> anyhow::Result<texture::Texture> {
    println!("trying to load {:?}", file_name);
    let data = load_binary(file_name).await?;
    texture::Texture::from_bytes(
        device,
        queue,
        &data,
        file_name,
        is_normal_map,
        mag_filter_mode,
    )
}

pub async fn load_binary(file_name: &str) -> anyhow::Result<Vec<u8>> {
    let path = std::path::Path::new::<PathBuf>(&env::current_dir()?.into())
        .join("res")
        .join(file_name);
    let data = std::fs::read(path)?;

    Ok(data)
}

pub async fn load_string(file_name: &str) -> anyhow::Result<String> {
    let path = std::path::Path::new::<PathBuf>(&env::current_dir()?.into())
        .join("res")
        .join(file_name);
    //print!("{:?}", path);
    let p = &path;
    if let Err(err) = std::fs::read_to_string(p) {
        println!("couldnt find {:?}", p);
        eprintln!("ERROR: {}", err);
        std::process::exit(1);
    }
    Ok(std::fs::read_to_string(path)?)
}

pub async fn load_cube(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<model::Model> {
    let mut materials = Vec::new();
    let mut meshes = Vec::new();

    let top_texture = load_texture(
        "grass_block_top.png",
        false,
        device,
        queue,
        wgpu::FilterMode::Nearest,
    )
    .await?;
    let down_texture =
        load_texture("dirt.png", false, device, queue, wgpu::FilterMode::Nearest).await?;
    let side_texture = load_texture(
        "grass_block_side.png",
        false,
        device,
        queue,
        wgpu::FilterMode::Nearest,
    )
    .await?;
    let top_normal = load_texture(
        "grass_block_top_n.png",
        true,
        device,
        queue,
        wgpu::FilterMode::Nearest,
    )
    .await?;
    let down_normal =
        load_texture("dirt_n.png", true, device, queue, wgpu::FilterMode::Nearest).await?;
    let side_normal = load_texture(
        "grass_block_side_n.png",
        true,
        device,
        queue,
        wgpu::FilterMode::Nearest,
    )
    .await?;

    materials.push(model::Material::new(
        device,
        "top material",
        top_texture,
        top_normal,
        layout,
    ));
    materials.push(model::Material::new(
        device,
        "down material",
        down_texture,
        down_normal,
        layout,
    ));
    materials.push(model::Material::new(
        device,
        "side material",
        side_texture,
        side_normal,
        layout,
    ));

    let mut CUBE_FRONT: &mut [ModelVertex] = &mut [
        // FACING FRONT
        ModelVertex {
            position: [-0.5, 0.5, 0.5],
            tex_coords: [0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            vertex_color: [1.0, 1.0, 1.0],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
        ModelVertex {
            position: [-0.5, -0.5, 0.5],
            tex_coords: [0.0, 1.0],
            normal: [0.0, 0.0, 1.0],
            vertex_color: [1.0, 1.0, 1.0],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
        ModelVertex {
            position: [0.5, -0.5, 0.5],
            tex_coords: [1.0, 1.0],
            normal: [0.0, 0.0, 1.0],
            vertex_color: [1.0, 1.0, 1.0],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
        ModelVertex {
            position: [0.5, 0.5, 0.5],
            tex_coords: [1.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            vertex_color: [1.0, 1.0, 1.0],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
    ];

    let mut CUBE_UP: &mut [ModelVertex] = &mut [
        // FACING UP
        ModelVertex {
            position: [-0.5, 0.5, -0.5],
            tex_coords: [0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            vertex_color: [0.24620132670783548, 0.8069522576692516, 0.13843161503245183],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
        ModelVertex {
            position: [-0.5, 0.5, 0.5],
            tex_coords: [0.0, 1.0],
            normal: [0.0, 1.0, 0.0],
            vertex_color: [0.24620132670783548, 0.8069522576692516, 0.13843161503245183],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
        ModelVertex {
            position: [0.5, 0.5, 0.5],
            tex_coords: [1.0, 1.0],
            normal: [0.0, 1.0, 0.0],
            vertex_color: [0.24620132670783548, 0.8069522576692516, 0.13843161503245183],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
        ModelVertex {
            position: [0.5, 0.5, -0.5],
            tex_coords: [1.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            vertex_color: [0.24620132670783548, 0.8069522576692516, 0.13843161503245183],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
    ];

    let mut CUBE_LEFT: &mut [ModelVertex] = &mut [
        // FACING LEFT
        ModelVertex {
            position: [-0.5, 0.5, -0.5],
            tex_coords: [0.0, 0.0],
            normal: [-1.0, 0.0, 0.0],
            vertex_color: [1.0, 1.0, 1.0],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
        ModelVertex {
            position: [-0.5, -0.5, -0.5],
            tex_coords: [0.0, 1.0],
            normal: [-1.0, 0.0, 0.0],
            vertex_color: [1.0, 1.0, 1.0],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
        ModelVertex {
            position: [-0.5, -0.5, 0.5],
            tex_coords: [1.0, 1.0],
            normal: [-1.0, 0.0, 0.0],
            vertex_color: [1.0, 1.0, 1.0],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
        ModelVertex {
            position: [-0.5, 0.5, 0.5],
            tex_coords: [1.0, 0.0],
            normal: [-1.0, 0.0, 0.0],
            vertex_color: [1.0, 1.0, 1.0],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
    ];

    let mut CUBE_RIGHT: &mut [ModelVertex] = &mut [
        // FACING RIGHT
        ModelVertex {
            position: [0.5, 0.5, 0.5],
            tex_coords: [0.0, 0.0],
            normal: [1.0, 0.0, 0.0],
            vertex_color: [1.0, 1.0, 1.0],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
        ModelVertex {
            position: [0.5, -0.5, 0.5],
            tex_coords: [0.0, 1.0],
            normal: [1.0, 0.0, 0.0],
            vertex_color: [1.0, 1.0, 1.0],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
        ModelVertex {
            position: [0.5, -0.5, -0.5],
            tex_coords: [1.0, 1.0],
            normal: [1.0, 0.0, 0.0],
            vertex_color: [1.0, 1.0, 1.0],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
        ModelVertex {
            position: [0.5, 0.5, -0.5],
            tex_coords: [1.0, 0.0],
            normal: [1.0, 0.0, 0.0],
            vertex_color: [1.0, 1.0, 1.0],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
    ];

    let CUBE_BACK: &mut [ModelVertex] = &mut [
        // FACING BACK
        ModelVertex {
            position: [0.5, 0.5, -0.5],
            tex_coords: [0.0, 0.0],
            normal: [0.0, 0.0, -1.0],
            vertex_color: [1.0, 1.0, 1.0],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
        ModelVertex {
            position: [0.5, -0.5, -0.5],
            tex_coords: [0.0, 1.0],
            normal: [0.0, 0.0, -1.0],
            vertex_color: [1.0, 1.0, 1.0],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
        ModelVertex {
            position: [-0.5, -0.5, -0.5],
            tex_coords: [1.0, 1.0],
            normal: [0.0, 0.0, -1.0],
            vertex_color: [1.0, 1.0, 1.0],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
        ModelVertex {
            position: [-0.5, 0.5, -0.5],
            tex_coords: [1.0, 0.0],
            normal: [0.0, 0.0, -1.0],
            vertex_color: [1.0, 1.0, 1.0],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
    ];

    let mut CUBE_DOWN: &mut [ModelVertex] = &mut [
        // FACING DOWN
        ModelVertex {
            position: [-0.5, -0.5, 0.5],
            tex_coords: [0.0, 0.0],
            normal: [0.0, -1.0, 0.0],
            vertex_color: [1.0, 1.0, 1.0],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
        ModelVertex {
            position: [-0.5, -0.5, -0.5],
            tex_coords: [0.0, 1.0],
            normal: [0.0, -1.0, 0.0],
            vertex_color: [1.0, 1.0, 1.0],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
        ModelVertex {
            position: [0.5, -0.5, -0.5],
            tex_coords: [1.0, 1.0],
            normal: [0.0, -1.0, 0.0],
            vertex_color: [1.0, 1.0, 1.0],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
        ModelVertex {
            position: [0.5, -0.5, 0.5],
            tex_coords: [1.0, 0.0],
            normal: [0.0, -1.0, 0.0],
            vertex_color: [1.0, 1.0, 1.0],
            tangent: [0.0; 3],
            bitangent: [0.0; 3],
        },
    ];

    let mut CUBE_FACES: &mut [&mut [ModelVertex]] = &mut [
        CUBE_UP, CUBE_DOWN, CUBE_LEFT, CUBE_RIGHT, CUBE_FRONT, CUBE_BACK,
    ];

    let mut FACE_INDICES: &mut [u32] = &mut [0, 1, 2, 2, 3, 0];

    let mut CUBE_INDICES2: &mut [[u32; 6]; 6] = &mut [
        [0, 2, 3, 3, 1, 0],
        [1, 3, 7, 7, 5, 1],
        [5, 7, 6, 6, 4, 5],
        [4, 0, 1, 1, 5, 4],
        [2, 6, 7, 7, 3, 2],
        [4, 6, 2, 2, 0, 4],
    ];

    for i in 0..6 {
        let vertices: &mut [ModelVertex] = &mut CUBE_FACES[i];
        let indices = CUBE_INDICES2[i];
        let mut triangles_included = vec![0; 4];

        for c in FACE_INDICES.chunks(3) {
            let v0 = vertices[c[0] as usize];
            let v1 = vertices[c[1] as usize];
            let v2 = vertices[c[2] as usize];

            let pos0: cgmath::Vector3<_> = v0.position.into();
            let pos1: cgmath::Vector3<_> = v1.position.into();
            let pos2: cgmath::Vector3<_> = v2.position.into();

            let uv0: cgmath::Vector2<_> = v0.tex_coords.into();
            let uv1: cgmath::Vector2<_> = v1.tex_coords.into();
            let uv2: cgmath::Vector2<_> = v2.tex_coords.into();

            // Calculate the edges of the triangle
            let delta_pos1 = pos1 - pos0;
            let delta_pos2 = pos2 - pos0;

            // This will give us a direction to calculate the
            // tangent and bitangent
            let delta_uv1 = uv1 - uv0;
            let delta_uv2 = uv2 - uv0;

            // Solving the following system of equations will
            // give us the tangent and bitangent.
            //     delta_pos1 = delta_uv1.x * T + delta_uv1.y * B
            //     delta_pos2 = delta_uv2.x * T + delta_uv2.y * B
            // Luckily, the place I found this equation provided
            // the solution!
            let r = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);
            let tangent = (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r;
            // We flip the bitangent to enable right-handed normal
            // maps with wgpu texture coordinate system
            let bitangent = (delta_pos2 * delta_uv1.x - delta_pos1 * delta_uv2.x) * -r;

            // We'll use the same tangent/bitangent for each vertex in the triangle
            vertices[c[0] as usize].tangent =
                (tangent + cgmath::Vector3::from(vertices[c[0] as usize].tangent)).into();
            vertices[c[1] as usize].tangent =
                (tangent + cgmath::Vector3::from(vertices[c[1] as usize].tangent)).into();
            vertices[c[2] as usize].tangent =
                (tangent + cgmath::Vector3::from(vertices[c[2] as usize].tangent)).into();
            vertices[c[0] as usize].bitangent =
                (bitangent + cgmath::Vector3::from(vertices[c[0] as usize].bitangent)).into();
            vertices[c[1] as usize].bitangent =
                (bitangent + cgmath::Vector3::from(vertices[c[1] as usize].bitangent)).into();
            vertices[c[2] as usize].bitangent =
                (bitangent + cgmath::Vector3::from(vertices[c[2] as usize].bitangent)).into();

            // Used to average the tangents/bitangents
            triangles_included[c[0] as usize] += 1;
            triangles_included[c[1] as usize] += 1;
            triangles_included[c[2] as usize] += 1;
        }

        // Average the tangents/bitangents
        for (i, n) in triangles_included.into_iter().enumerate() {
            let denom = 1.0 / n as f32;
            let v = &mut vertices[i];
            v.tangent = (cgmath::Vector3::from(v.tangent) * denom).into();
            v.bitangent = (cgmath::Vector3::from(v.bitangent) * denom).into();
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{:?} Vertex Buffer", "cube")),
            contents: bytemuck::cast_slice(CUBE_FACES[i]),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{:?} Index Buffer", "cube")),
            contents: bytemuck::cast_slice(&FACE_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        meshes.push(Mesh {
            name: "cube".to_owned(),
            vertex_buffer,
            index_buffer,
            num_elements: 6 as u32,
            material: match i {
                0 => 0, // up
                1 => 1, // down
                2 => 2, // left
                3 => 2, // right
                4 => 2, // front
                5 => 2, // back
                _ => panic!("too many faces in a cube"),
            },
        });
    }

    Ok(model::Model { meshes, materials })
}

pub async fn load_model(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<model::Model> {
    let obj_text = load_string(file_name).await?;
    let obj_cursor = Cursor::new(obj_text);
    let mut obj_reader = BufReader::new(obj_cursor);

    let (models, obj_materials) = tobj::load_obj_buf_async(
        &mut obj_reader,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
        |p| async move {
            println!("about to load {}", p);
            let mat_text = load_string(&p).await.unwrap();

            tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
        },
    )
    .await?;

    println!("beginning materials");

    let mut materials = Vec::new();
    if let Err(err) = obj_materials {
        println!("error loading materials");
        eprintln!("ERROR: {}", err);
        std::process::exit(1);
    }
    for m in obj_materials? {
        let res = load_texture(
            &m.diffuse_texture,
            false,
            device,
            queue,
            wgpu::FilterMode::Linear,
        )
        .await;
        if let Err(err) = res {
            println!("could not find {:?}", m.diffuse_texture);
            eprintln!("ERROR: {}", err);
            std::process::exit(1);
        }
        let diffuse_texture = res?;

        let res_normal = load_texture(
            &m.normal_texture,
            true,
            device,
            queue,
            wgpu::FilterMode::Linear,
        )
        .await;
        if let Err(err) = res_normal {
            println!("could not find {:?}", m.normal_texture);
            eprintln!("ERROR: {}", err);
            std::process::exit(1);
        }
        let normal_texture = res_normal?;

        materials.push(model::Material::new(
            device,
            &m.name,
            diffuse_texture,
            normal_texture,
            layout,
        ));
    }

    let meshes = models
        .into_iter()
        .map(|m| {
            let mut vertices = (0..m.mesh.positions.len() / 3)
                .map(|i| model::ModelVertex {
                    position: [
                        m.mesh.positions[i * 3],
                        m.mesh.positions[i * 3 + 1],
                        m.mesh.positions[i * 3 + 2],
                    ],
                    tex_coords: [m.mesh.texcoords[i * 2], 1.0 - m.mesh.texcoords[i * 2 + 1]],
                    normal: [
                        m.mesh.normals[i * 3],
                        m.mesh.normals[i * 3 + 1],
                        m.mesh.normals[i * 3 + 2],
                    ],
                    tangent: [0.0; 3],
                    bitangent: [0.0; 3],
                    vertex_color: [1.0; 3],
                })
                .collect::<Vec<_>>();

            let indices = &m.mesh.indices;
            let mut triangles_included = vec![0; vertices.len()];

            // Calculate tangents and bitangets. We're going to
            // use the triangles, so we need to loop through the
            // indices in chunks of 3

            for c in indices.chunks(3) {
                let v0 = vertices[c[0] as usize];
                let v1 = vertices[c[1] as usize];
                let v2 = vertices[c[2] as usize];

                let pos0: cgmath::Vector3<_> = v0.position.into();
                let pos1: cgmath::Vector3<_> = v1.position.into();
                let pos2: cgmath::Vector3<_> = v2.position.into();

                let uv0: cgmath::Vector2<_> = v0.tex_coords.into();
                let uv1: cgmath::Vector2<_> = v1.tex_coords.into();
                let uv2: cgmath::Vector2<_> = v2.tex_coords.into();

                // Calculate the edges of the triangle
                let delta_pos1 = pos1 - pos0;
                let delta_pos2 = pos2 - pos0;

                // This will give us a direction to calculate the
                // tangent and bitangent
                let delta_uv1 = uv1 - uv0;
                let delta_uv2 = uv2 - uv0;

                // Solving the following system of equations will
                // give us the tangent and bitangent.
                //     delta_pos1 = delta_uv1.x * T + delta_uv1.y * B
                //     delta_pos2 = delta_uv2.x * T + delta_uv2.y * B
                // Luckily, the place I found this equation provided
                // the solution!
                let r = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);
                let tangent = (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r;
                // We flip the bitangent to enable right-handed normal
                // maps with wgpu texture coordinate system
                let bitangent = (delta_pos2 * delta_uv1.x - delta_pos1 * delta_uv2.x) * -r;

                // We'll use the same tangent/bitangent for each vertex in the triangle
                vertices[c[0] as usize].tangent =
                    (tangent + cgmath::Vector3::from(vertices[c[0] as usize].tangent)).into();
                vertices[c[1] as usize].tangent =
                    (tangent + cgmath::Vector3::from(vertices[c[1] as usize].tangent)).into();
                vertices[c[2] as usize].tangent =
                    (tangent + cgmath::Vector3::from(vertices[c[2] as usize].tangent)).into();
                vertices[c[0] as usize].bitangent =
                    (bitangent + cgmath::Vector3::from(vertices[c[0] as usize].bitangent)).into();
                vertices[c[1] as usize].bitangent =
                    (bitangent + cgmath::Vector3::from(vertices[c[1] as usize].bitangent)).into();
                vertices[c[2] as usize].bitangent =
                    (bitangent + cgmath::Vector3::from(vertices[c[2] as usize].bitangent)).into();

                // Used to average the tangents/bitangents
                triangles_included[c[0] as usize] += 1;
                triangles_included[c[1] as usize] += 1;
                triangles_included[c[2] as usize] += 1;
            }

            // Average the tangents/bitangents
            for (i, n) in triangles_included.into_iter().enumerate() {
                let denom = 1.0 / n as f32;
                let v = &mut vertices[i];
                v.tangent = (cgmath::Vector3::from(v.tangent) * denom).into();
                v.bitangent = (cgmath::Vector3::from(v.bitangent) * denom).into();
            }

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Vertex Buffer", file_name)),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Index Buffer", file_name)),
                contents: bytemuck::cast_slice(&m.mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            model::Mesh {
                name: file_name.to_string(),
                vertex_buffer,
                index_buffer,
                num_elements: m.mesh.indices.len() as u32,
                material: m.mesh.material_id.unwrap_or(0),
            }
        })
        .collect::<Vec<_>>();

    Ok(model::Model { meshes, materials })
}

pub struct HdrLoader {
    texture_format: wgpu::TextureFormat,
    equirect_layout: wgpu::BindGroupLayout,
    equirect_to_cubemap: wgpu::ComputePipeline,
}

impl HdrLoader {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("equirectangular.wgsl"));
        let texture_format = wgpu::TextureFormat::Rgba32Float;
        let equirect_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HdrLoader::equirect_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: texture_format,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&equirect_layout],
            push_constant_ranges: &[],
        });

        let equirect_to_cubemap =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("equirect_to_cubemap"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("compute_equirect_to_cubemap"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            equirect_to_cubemap,
            texture_format,
            equirect_layout,
        }
    }

    pub fn from_equirectangular_bytes(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
        dst_size: u32,
        label: Option<&str>,
    ) -> anyhow::Result<texture::CubeTexture> {
        let hdr_decoder = HdrDecoder::new(Cursor::new(data))?;
        let meta = hdr_decoder.metadata();

        #[cfg(not(target_arch = "wasm32"))]
        let pixels = {
            let mut pixels = vec![[0.0, 0.0, 0.0, 0.0]; meta.width as usize * meta.height as usize];
            hdr_decoder.read_image_transform(
                |pix| {
                    let rgb = pix.to_hdr();
                    [rgb.0[0], rgb.0[1], rgb.0[2], 1.0f32]
                },
                &mut pixels[..],
            )?;
            pixels
        };
        #[cfg(target_arch = "wasm32")]
        let pixels = hdr_decoder
            .read_image_native()?
            .into_iter()
            .map(|pix| {
                let rgb = pix.to_hdr();
                [rgb.0[0], rgb.0[1], rgb.0[2], 1.0f32]
            })
            .collect::<Vec<_>>();

        let src = texture::Texture::create_2d_texture(
            device,
            meta.width,
            meta.height,
            self.texture_format,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            wgpu::FilterMode::Linear,
            None,
        );

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &src.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytemuck::cast_slice(&pixels),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(src.size.width * std::mem::size_of::<[f32; 4]>() as u32),
                rows_per_image: Some(src.size.height),
            },
            src.size,
        );

        let dst = texture::CubeTexture::create_2d(
            device,
            dst_size,
            dst_size,
            self.texture_format,
            1,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            wgpu::FilterMode::Nearest,
            label,
        );

        let dst_view = dst.texture().create_view(&wgpu::TextureViewDescriptor {
            label,
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            // array_layer_count: Some(6),
            ..Default::default()
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label,
            layout: &self.equirect_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&src.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&dst_view),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label,
            timestamp_writes: None,
        });

        let num_workgroups = (dst_size + 15) / 16;
        pass.set_pipeline(&self.equirect_to_cubemap);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_workgroups, num_workgroups, 6);

        drop(pass);

        queue.submit([encoder.finish()]);

        Ok(dst)
    }
}
