#![allow(dead_code)]
#![allow(unused_variables)]
extern crate glm;
extern crate rayon;
extern crate stb_image;

use rayon::prelude::*;
use std::cmp;
use std::rc::Rc;
use std::sync::Arc;
use unsafe_unwrap::UnsafeUnwrap;

pub type Vec3 = glm::Vector3<f32>;
pub type Vec4 = glm::Vector4<f32>;
pub type Vec2 = glm::Vector2<f32>;
pub use glm::vec2;
pub use glm::vec3;
pub use glm::vec4;

pub type Vec2i = glm::Vector2<i32>;
pub type Vec3i = glm::Vector3<i32>;
pub type Vec4i = glm::Vector4<i32>;
pub use glm::ivec2 as vec2i;
pub use glm::ivec3 as vec3i;
pub use glm::ivec4 as vec4i;

pub use glm::builtin::ceil;
pub use glm::builtin::clamp_s;
pub use glm::builtin::floor;
pub use glm::builtin::fract;
pub use glm::builtin::mix as lerp;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

pub struct Texture2D {
    pub data: Vec<f32>, // 4-bytes aligned
    pub width: u32,
    pub height: u32,
    pub channel: u8,
}

impl Default for Texture2D {
    fn default() -> Self {
        Texture2D {
            data: vec![0.0f32; 0],
            width: 0,
            height: 0,
            channel: 0,
        }
    }
}

impl Texture2D {
    pub fn from_integer_data(data: Vec<u8>, width: u32, height: u32, channel: u8) -> Self {
        Texture2D {
            data: data.iter().map(|&v| v as f32).collect(),
            width: width,
            height: height,
            channel: channel,
        }
    }
    pub fn from_float_data(data: Vec<f32>, width: u32, height: u32, channel: u8) -> Self {
        Texture2D {
            data: data,
            width: width,
            height: height,
            channel: channel,
        }
    }
    pub fn sample_nearest(&self, x: f32, y: f32) -> Vec3 {
        let fx = floor(x * self.width as f32) as u32;
        let fy = floor(y * self.height as f32) as u32;
        let idx = ((fx + fy * self.width) * self.channel as u32) as usize;
        vec3(self.data[idx], self.data[idx + 1], self.data[idx + 2])
    }

    pub fn raw_data(&self) -> &Vec<f32> {
        &self.data
    }

    pub fn raw_data_mut(&mut self) -> &mut Vec<f32> {
        &mut self.data
    }
}

#[derive(Copy, Clone, Debug)]
struct Bound2i {
    pub min: Vec2i,
    pub max: Vec2i,
}

impl Default for Bound2i {
    fn default() -> Self {
        Self::new()
    }
}

// unsafe impl std::marker::Sync for Bound2i{}
// unsafe impl std::marker::Send for Bound2i{}

impl Bound2i {
    pub fn new() -> Self {
        Bound2i {
            min: vec2i(std::i32::MAX, std::i32::MAX),
            max: vec2i(std::i32::MIN, std::i32::MIN),
        }
    }
    pub fn from_points(p0: &Vec2i, p1: &Vec2i) -> Self {
        Bound2i {
            min: vec2i(cmp::min(p0.x, p1.x), cmp::min(p0.y, p1.y)),
            max: vec2i(cmp::max(p0.x, p1.x), cmp::max(p0.y, p1.y)),
        }
    }
    pub fn null(&self) -> bool {
        !(self.max.x > self.min.x && self.max.y > self.min.y)
    }
    pub fn union(&mut self, p: &Vec2i) {
        self.min = vec2i(cmp::min(self.min.x, p.x), cmp::min(self.min.y, p.y));
        self.max = vec2i(cmp::max(self.max.x, p.x), cmp::max(self.max.y, p.y));
    }
    pub fn intersect(&mut self, other: &Bound2i) -> Self {
        Bound2i {
            min: vec2i(
                cmp::max(self.min.x, other.min.x),
                cmp::max(self.min.y, other.min.y),
            ),
            max: vec2i(
                cmp::min(self.max.x, other.max.x),
                cmp::min(self.max.y, other.max.y),
            ),
        }
    }
}

#[inline]
fn barycentric(abc: &[Vec2i; 3], p: &Vec2i) -> Vec3 {
    let a = vec3(
        (abc[2].x - abc[0].x) as f32,
        (abc[1].x - abc[0].x) as f32,
        (abc[0].x - p.x) as f32,
    );
    let b = vec3(
        (abc[2].y - abc[0].y) as f32,
        (abc[1].y - abc[0].y) as f32,
        (abc[0].y - p.y) as f32,
    );
    let c = glm::cross(a, b);
    if c.z.abs() < 1.0f32 {
        vec3(-1.0f32, 1.0f32, 1.0f32)
    } else {
        vec3(1.0f32 - (c.x + c.y) / c.z, c.y / c.z, c.x / c.z)
    }
}

#[allow(non_snake_case, non_camel_case_types)]
pub struct VS_IN<'a> {
    pub vertex: &'a Vec3,
    pub texCoord: &'a Vec2,
    pub norm: &'a Vec3,
}

#[allow(non_snake_case)]
pub struct PerVertAttrib {
    pub vertices: Vec<Vec3>,
    pub texCoords: Vec<Vec2>,
    pub norms: Vec<Vec3>,
}

impl Default for PerVertAttrib {
    fn default() -> Self {
        PerVertAttrib {
            vertices: vec![vec3(0., 0., 0.); 0],
            texCoords: vec![vec2(0., 0.); 0],
            norms: vec![vec3(0., 0., 0.); 0],
        }
    }
}

impl PerVertAttrib {
    pub fn new() -> Self {
        PerVertAttrib {
            vertices: vec![vec3(0., 0., 0.); 0],
            texCoords: vec![vec2(0., 0.); 0],
            norms: vec![vec3(0., 0., 0.); 0],
        }
    }
}

#[allow(non_camel_case_types, non_snake_case)]
pub struct VS_OUT_FS_IN {
    pub vertex: Vec3,
    pub texCoord: Vec2,
    pub norm: Vec3,
}

impl Default for VS_OUT_FS_IN {
    fn default() -> Self {
        Self {
            vertex: vec3(0.0, 0.0, 0.0),
            texCoord: vec2(0.0, 0.0),
            norm: vec3(0.0, 0.0, 0.0),
        }
    }
}

#[allow(non_camel_case_types)]
pub struct FS_OUT {
    pub color: Vec4,
}

impl Default for FS_OUT {
    fn default() -> Self {
        Self {
            color: vec4(0.0, 0.0, 0.0, 0.0),
        }
    }
}

#[allow(non_camel_case_types)]
pub struct Refresher<'a> {
    resolution: (u32, u32),
    screenbox: Bound2i,
    buffer_bytes: usize,
    pixel_count: usize,
    color_component: u32,
    clearcolor: Color,
    enablefaceculling: bool,
    def_fb: Vec<u8>,
    def_db: Vec<f32>,
    framebuffer: *mut u8,
    depthbuffer: *mut f32,
    pervertexattrib: PerVertAttrib,
    indexbuffer: Option<&'a Vec<i32>>,
    vertexshader: Option<Arc<dyn Fn(&VS_IN, &mut VS_OUT_FS_IN) -> Vec4 + 'a>>,
    fragmentshader: Option<Arc<dyn Fn(&VS_OUT_FS_IN, &mut FS_OUT) -> bool + 'a>>,
}

unsafe impl<'a> std::marker::Sync for Refresher<'a> {}
unsafe impl<'a> std::marker::Send for Refresher<'a> {}

#[allow(non_camel_case_types)]
impl<'a> Refresher<'a> {
    pub fn new(res: (u32, u32)) -> Self {
        let comp = 4;
        let buffer_size = res.0 as usize * res.1 as usize * comp as usize;
        let aabb = Bound2i {
            min: vec2i(0, 0),
            max: vec2i(res.0 as i32, res.1 as i32),
        };

        let mut refresher = Refresher {
            resolution: res,
            screenbox: aabb,
            buffer_bytes: buffer_size,
            pixel_count: res.0 as usize * res.1 as usize,
            color_component: comp,
            clearcolor: Color {
                r: 0,
                g: 0,
                b: 0,
                a: 0,
            },
            enablefaceculling: false,
            def_fb: vec![0u8; buffer_size as usize],
            def_db: vec![1.0f32; buffer_size],
            framebuffer: std::ptr::null_mut(),
            depthbuffer: std::ptr::null_mut(),
            pervertexattrib: PerVertAttrib::new(),
            indexbuffer: None,
            vertexshader: None,
            fragmentshader: None,
        };
        refresher.framebuffer = refresher.def_fb.as_mut_ptr();
        refresher.depthbuffer = refresher.def_db.as_mut_ptr();
        refresher
    }

    pub fn buffer_bytes(&self) -> usize {
        self.buffer_bytes
    }
    pub fn color_component(&self) -> u32 {
        self.color_component
    }

    pub fn set_clear_color(&mut self, clearcolor: Color) {
        self.clearcolor = clearcolor;
    }

    pub fn enable_face_culling(&mut self, enable: bool) {
        self.enablefaceculling = enable;
    }

    pub fn set_per_vertex_attribute(&mut self, attrib: PerVertAttrib) {
        self.pervertexattrib = attrib;
    }

    pub fn resolution(&self) -> (u32, u32) {
        self.resolution
    }

    pub fn set_vertex_shader<F>(&mut self, vs: F)
    where
        F: Fn(&VS_IN, &mut VS_OUT_FS_IN) -> Vec4 + 'a,
    {
        self.vertexshader = Some(Arc::new(vs));
    }

    pub fn set_fragment_shader<F>(&mut self, fs: F)
    where
        F: Fn(&VS_OUT_FS_IN, &mut FS_OUT) -> bool + 'a,
    {
        self.fragmentshader = Some(Arc::new(fs));
    }

    pub fn set_index(&mut self, index: &'a Vec<i32>) {
        self.indexbuffer = Some(index);
    }

    fn rasterize(&self, i: usize, vid: i32) {
        // warning: The camera and vertex are in the same plane will cause v.w == 0,
        // we don't validate it here

        let ib = self.indexbuffer.as_ref().unwrap();
        let vs = self.vertexshader.as_ref().unwrap().clone();
        let fs = self.fragmentshader.as_ref().unwrap().clone();

        // per-vertex attribute
        let vb = &self.pervertexattrib.vertices;
        let tb = &self.pervertexattrib.texCoords;
        let nb = &self.pervertexattrib.norms;

        let i0 = ib[i] as usize;
        let i1 = ib[i + 1] as usize;
        let i2 = ib[i + 2] as usize;

        let p0_vs_in = VS_IN {
            vertex: &vb[i0],
            texCoord: &tb[i0],
            norm: &nb[i0],
        };

        let p1_vs_in = VS_IN {
            vertex: &vb[i1],
            texCoord: &tb[i1],
            norm: &nb[i1],
        };

        let p2_vs_in = VS_IN {
            vertex: &vb[i2],
            texCoord: &tb[i2],
            norm: &nb[i2],
        };

        let mut p0_vs_out: VS_OUT_FS_IN = Default::default();
        let mut p1_vs_out: VS_OUT_FS_IN = Default::default();
        let mut p2_vs_out: VS_OUT_FS_IN = Default::default();

        let cp0 = vs(&p0_vs_in, &mut p0_vs_out);
        let cp1 = vs(&p1_vs_in, &mut p1_vs_out);
        let cp2 = vs(&p2_vs_in, &mut p2_vs_out);

        if self.enablefaceculling
            && glm::cross(
                cp2.truncate(3) - cp0.truncate(3),
                cp1.truncate(3) - cp0.truncate(3),
            )
            .z > 0.0
        {
            // Culls face
            return;
        }

        let to_screen = |v: &Vec4| -> (Vec2i, f32) {
            let x = ((v.x + 1.0) / 2.0 * (self.resolution.0 as f32)) as i32;
            let y = ((v.y + 1.0) / 2.0 * (self.resolution.1 as f32)) as i32;
            (vec2i(x, self.resolution.1 as i32 - y), v.z)
        };

        let persp_div = |v: &Vec4| -> Vec4 { *v / (*v).w };

        let p0 = to_screen(&persp_div(&cp0));
        let p1 = to_screen(&persp_div(&cp1));
        let p2 = to_screen(&persp_div(&cp2));

        let mut aabb = Bound2i::from_points(&p0.0, &p1.0);
        aabb.union(&p2.0);
        aabb = aabb.intersect(&self.screenbox); // clip

        for x in aabb.min.x..aabb.max.x {
            for y in aabb.min.y..aabb.max.y {
                let p = vec2i(x, y);
                let res = barycentric(&[p0.0, p1.0, p2.0], &p); // Check the current pixel is in the triangle by the barycentric
                if res.x < 0.0f32 || res.y < 0.0f32 || res.z < 0.0f32 {
                    continue;
                }
                let d = p0.1 * res.x + p1.1 * res.y + p2.1 * res.z; // interpulate the depth
                let interp_vertex =
                    p0_vs_out.vertex * res.x + p1_vs_out.vertex * res.y + p2_vs_out.vertex * res.z;
                let interp_norm =
                    p0_vs_out.norm * res.x + p1_vs_out.norm * res.y + p2_vs_out.norm * res.z;
                let interp_tex = p0_vs_out.texCoord * res.x
                    + p1_vs_out.texCoord * res.y
                    + p2_vs_out.texCoord * res.z;

                let fs_in = VS_OUT_FS_IN {
                    vertex: interp_vertex,
                    texCoord: interp_tex,
                    norm: interp_norm,
                };

                let mut fs_out: FS_OUT = Default::default();

                let not_discard = fs(&fs_in, &mut fs_out);
                let index = (x as usize + y as usize * self.resolution.0 as usize) as usize;

                //let db = (*((&self.def_db) as *const Vec<f32> as *mut Vec<f32>)).as_mut_slice();
                // if *db.get_unchecked_mut(index) > d && not_discard {
                // *db.get_unchecked_mut(index) = d;
                // self.set_pixel_unsafe(index, &fs_out.color);
                // }
                if self.get_depth_value(index) > d && not_discard {
                    self.set_depth_value(index, d);
                    self.set_pixel_unsafe(index, &fs_out.color);
                }
            }
        }
    }

    pub fn refresh(&self) {
        let ib = self.indexbuffer.as_ref().unwrap();
        ib.par_iter()
            .enumerate()
            .step_by(3)
            .for_each(|(i, &vid)| self.rasterize(i, vid));
    }

    pub fn raw_buffer(&self) -> &Vec<u8> {
        &self.def_fb
    }

    pub fn raw_buffer_mut(&mut self) -> &mut Vec<u8> {
        &mut self.def_fb
    }

    pub fn raw_depth_buffer(&self) -> &Vec<f32> {
        &self.def_db
    }

    pub fn clear_color_buffer(&mut self) {
        let pixels = self.buffer_bytes / self.color_component as usize;
        //let comp = self.color_component as usize;
        (0..pixels).into_par_iter().for_each(|pixel_index|{
            self.set_pixel_unsafe_2(pixel_index, &self.clearcolor);
        });
        // unsafe{
            // std::ptr::copy(self.cleared_fb.as_ptr(),self.framebuffer,self.cleared_fb.len());
        // }
    }

     fn set_pixel_unsafe_2(&self, pixel_index: usize, color: &Color) {
        unsafe {
            let comp = self.color_component as isize;
            let idx = pixel_index as isize;
            *self.framebuffer.offset(comp * idx) = color.r;
            *self.framebuffer.offset(comp * idx + 1) = color.g;
            *self.framebuffer.offset(comp * idx + 2) = color.b;
            *self.framebuffer.offset(comp * idx + 3) = color.a;
        }
        //let fb = (*self.fb_ptr).as_mut_slice();
    }

    fn set_pixel_unsafe(&self, pixel_index: usize, color: &Vec4) {
        unsafe {
            let fb = self.framebuffer;
            //let fb = (*self.fb_ptr).as_mut_slice();
            let comp = self.color_component as isize;
            let idx = pixel_index as isize;

            *self.framebuffer.offset(comp * idx) = (255.0 * color.x) as u8;
            *self.framebuffer.offset(comp * idx + 1) = (255.0 * color.y) as u8;
            *self.framebuffer.offset(comp * idx + 2) = (255.0 * color.z) as u8;
            *self.framebuffer.offset(comp * idx + 3) = (255.0 * color.w) as u8;
        }
    }
    fn set_depth_value(&self, pixel_index: usize, depth: f32) {
        unsafe {
            let db = self.depthbuffer;
            *db.offset(pixel_index as isize) = depth;
        }
    }
    fn get_depth_value(&self, pixel_index: usize) -> f32 {
        unsafe {
            let db = self.depthbuffer;
            let ret = *db.offset(pixel_index as isize);
            ret
        }
    }

    pub fn set_color_buffer(&mut self, fb: Option<&'a mut Vec<u8>>) {
        if let Some(buf) = fb {
            self.framebuffer = buf.as_mut_ptr();
        } else {
            self.framebuffer = self.def_fb.as_mut_ptr();
        }
    }

    pub fn set_depth_buffer(&mut self, db: Option<&'a mut Vec<f32>>) {
        if let Some(buf) = db {
            self.depthbuffer = buf.as_mut_ptr();
        } else {
            self.depthbuffer = self.def_db.as_mut_ptr();
        }
    }

    pub fn clear_depth_buffer(&mut self) {
        let pixels = self.buffer_bytes / self.color_component as usize;
        (0..pixels).into_par_iter().for_each(|pixel_index|{
            self.set_depth_value(pixel_index,1.0f32);
        });
    }
    pub fn set_pixel(&mut self, x: u32, y: u32, color: &Color) {
        let idx = (x + y * self.resolution.0) as isize;
        let comp = self.color_component as isize;
        unsafe {
            *self.framebuffer.offset(comp * idx) = color.r;
            *self.framebuffer.offset(comp * idx + 1) = color.g;
            *self.framebuffer.offset(comp * idx + 2) = color.b;
            *self.framebuffer.offset(comp * idx + 3) = color.a;
        }
    }
}
