#![allow(non_snake_case)]
extern crate assimp;
extern crate glm;
extern crate gltf;
extern crate sdl2;
use std::time::Instant;

use assimp::import::Importer;
use assimp::scene::Scene;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::PixelFormatEnum;
use sdl2::surface::Surface;

mod refresher;

use refresher::{vec2, vec3, vec4, Refresher, Vec2, Vec3, Vec4, FS_OUT, VS_IN, VS_OUT_FS_IN};

#[allow(dead_code)]
fn print_scene_info(scene: &Scene) {
    println!("Scene Info ----------------------");
    println!("Mesh Num: {}", scene.num_meshes());
    println!("Material Num: {}", scene.num_materials());
    println!("Texture Num: {}", scene.num_textures());
}

fn main() {
    let sdl_context = sdl2::init().unwrap();
    let mut event_pump = sdl_context.event_pump().unwrap();
    let vid = sdl_context.video().unwrap();

    let res: (u32, u32) = (800, 600);
    let window = vid
        .window("Rusterizer", res.0, res.1)
        .position_centered()
        .build()
        .unwrap();
    let mut canvas = window.into_canvas().build().unwrap();
    let texture_creator = canvas.texture_creator();

    // load asset
    let importer = Importer::new();
    let scene = importer
        .read_file("/home/ysl/Code/rusterizer/test/obj/diablo3_pose/diablo3_pose.obj")
        .unwrap();

    let mut vertex = Vec::<Vec3>::new();
    let mut tex_coord = Vec::<Vec2>::new();
    let mut norms = Vec::<Vec3>::new();
    let mut index = Vec::<i32>::new();

    for mesh in scene.mesh_iter() {
        vertex = mesh.vertex_iter().map(|v| vec3(v.x, v.y, v.z)).collect();
        tex_coord = mesh
            .texture_coords_iter(0)
            .map(|v| vec2(v.x, v.y))
            .collect();
        norms = mesh.normal_iter().map(|v| vec3(v.x, v.y, v.z)).collect();

        for f in mesh.face_iter() {
            let num = f.num_indices as isize;
            let indices = f.indices;
            for i in 0..num {
                unsafe {
                    index.push(*indices.offset(i) as i32);
                }
            }
        }
    }
    let tex_2d: refresher::Texture2D = match stb_image::image::load(
        "/home/ysl/Code/rusterizer/test/obj/diablo3_pose/diablo3_pose_diffuse.tga",
    ) {
        stb_image::image::LoadResult::Error(err) => {
            panic!("{}", err);
        }
        stb_image::image::LoadResult::ImageU8(image) => refresher::Texture2D::from_integer_data(
            image.data,
            image.width as u32,
            image.height as u32,
            image.depth as u8,
        ),
        stb_image::image::LoadResult::ImageF32(_image) => {
            panic!("float image is not supported");
        }
    };

    let attrib = refresher::PerVertAttrib {
        vertices: vertex,
        texCoords: tex_coord,
        norms: norms,
    };

    // create transformation
    let ident = glm::mat4(
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    );
    let s = 15.0f32;
    let camera_pos = vec3(15.0,15.0,15.0);
    let mut model = glm::ext::scale(&ident, vec3(s, s, s));
    let cam_center = vec3(0.0,0.0,0.0);
    let world_up = vec3(0.0,1.0,0.0);
    let view = glm::ext::look_at(
        camera_pos,
        cam_center,
        world_up
    );


    // create a light
    let light_pos = camera_pos;
    let light_target = cam_center;

    let light_view = glm::ext::look_at(light_pos,light_target,world_up);
    let proj = glm::ext::perspective(45.0f32, res.0 as f32 / res.0 as f32, 0.001f32, 1000.0f32);
    //let mvp = Arc::new(RwLock::new(proj * view * model));
    let mvp = proj * view * model;
    let light_mvp = proj * light_view * model;


    // create renderer
    let mut refresher = Refresher::new(res);

    // set vertex and index buffer
    //refresher.set_vertex(&vertex);
    
    refresher.set_per_vertex_attribute(attrib);
    refresher.set_index(&index);
    refresher.enable_face_culling(true);


    let depth_vs = |vs_in:&VS_IN,vs_out: &mut VS_OUT_FS_IN|->Vec4{
        let v = &vs_in.vertex;
        vs_out.vertex = *vs_in.vertex;
        vs_out.texCoord = *vs_in.texCoord;
        vs_out.norm = *vs_in.norm;
        light_mvp * vec4(v.x, v.y, v.z, 1.0)
    };

    let depth_fs = |fs_in: &VS_OUT_FS_IN, fs_out: &mut FS_OUT| -> bool {


        true
    };

    // create shaders
    let vs = |vs_in: &VS_IN, vs_out: &mut VS_OUT_FS_IN| -> Vec4 {
        let v = &vs_in.vertex;
        vs_out.vertex = *vs_in.vertex;
        vs_out.texCoord = *vs_in.texCoord;
        vs_out.norm = *vs_in.norm;
        mvp * vec4(v.x, v.y, v.z, 1.0)
    };

    let fs = |fs_in: &VS_OUT_FS_IN, fs_out: &mut FS_OUT| -> bool {
        use glm::builtin::{dot, max, normalize, pow};

        let frag_normal = normalize(fs_in.norm);
        let frag_pos = &fs_in.vertex;
        let eye_dir = normalize(camera_pos - *frag_pos);
        let light_dir = normalize(light_pos - *frag_pos);

        let H = (eye_dir + light_dir) / 2.0;

        let ambient = 0.2;
        let diffuse = max(dot(light_dir, frag_normal), 0.0);
        let spec = pow(max(dot(H, eye_dir), 0.0), 32.0);
        //let spec = max(dot(H,eye_dir),0.0);

        let tex_coord = fs_in.texCoord;
        let color = tex_2d.sample_nearest(tex_coord.x, tex_coord.y);
        fs_out.color = vec4(
            color.z as f32 / 255.0,
            color.y as f32 / 255.0,
            color.x as f32 / 255.0,
            1.0,
        ) * (ambient + diffuse + spec);
        true
    };

    refresher.set_vertex_shader(vs);
    refresher.set_fragment_shader(fs);

    let mut framecount = 0;
    let start = Instant::now();
    let mut prev = start.elapsed().as_secs_f32();

    // render loop
    'running: loop {
        // clear
        // update matrix
        //
        unsafe {
            model = glm::ext::rotate(&model, glm::radians(1.0), vec3(0.0, 1.0, 0.0));
            let mvp_r = &mvp as *const glm::Matrix4<f32> as *mut glm::Matrix4<f32>;
            *mvp_r = proj * view * model;
        }

        // rendering

        refresher.clear_depth_buffer();
        refresher.clear_color_buffer();
        refresher.refresh();
        framecount += 1;
        //start.elapsed().as_millis()

        if framecount % 200 == 0 {
            let cur = start.elapsed().as_secs_f32();
            let interval = cur - prev;
            prev = cur;
            println!(
                "FPS:{}, {} millis per frame \r\r",
                200.0f32 / interval,
                interval * 1000.0f32 / framecount as f32
            );
        }

        // create image from framebuffer and copy to the window
        let surface = Surface::from_data(
            refresher.raw_buffer_mut(),
            res.0,
            res.1,
            1,
            PixelFormatEnum::RGB888,
        )
        .unwrap();
        let texture = texture_creator
            .create_texture_from_surface(surface)
            .unwrap();
        canvas
            .copy(&texture, None, sdl2::rect::Rect::new(0, 0, res.0, res.1))
            .unwrap();
        canvas.present();

        // handling events
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asset_import() {
        use gltf::Gltf;
        let path = "/home/ysl/Downloads/gltf-models/2.0/Buggy/glTF/Buggy.gltf";
        let gltf = Gltf::open(path).unwrap();
        for scene in gltf.scenes() {
            for node in scene.nodes() {
                println!(
                    "Node {} has {} children",
                    node.index(),
                    node.children().count()
                );
                if let Some(m) = node.mesh() {
                    println!("mesh index:{}", m.index());
                    for p in m.primitives() {
                        println!("primitive");
                    }
                }
            }
        }

        let mut vertex = Vec::<Vec3>::new();
        let mut tex_coord = Vec::<Vec2>::new();
        let mut norms = Vec::<Vec3>::new();

        let (doc, buffers, images) = gltf::import(path).unwrap();
        for mesh in doc.meshes() {
            println!("Mesh #{}", mesh.index());
            for p in mesh.primitives() {
                println!("primitive #{}", p.index());
                let reader = p.reader(|buffer| Some(&buffers[buffer.index()]));
                if let Some(iter) = reader.read_positions() {
                    vertex = iter.map(|v| vec3(v[0], v[1], v[2])).collect();
                }
                if let Some(iter) = reader.read_normals() {
                    norms = iter.map(|v| vec3(v[0], v[1], v[2])).collect();
                }
                if let Some(iter) = reader.read_normals() {
                    norms = iter.map(|v| vec3(v[0], v[1], v[2])).collect();
                }
                if let Some(iter) = reader.read_tex_coords(0) {
                    tex_coord = iter.map(|v| vec2(v[0], v[1])).collect();
                }
            }
        }
    }
}
