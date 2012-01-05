"""
Copyright 2011 Ben Russell & contributors. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED ''AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL ANY
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of the contributors.
"""

import gc, random, time
from math import *
import ctypes

import pyglet
from pyglet import gl
from pyglet.gl import *

import numpy as np

import world
DRAW_DIST = 40

PHYSICS_SPF = 1.0/50.0
USE_EXPLICIT_LOOP = False

# set to True to pretend to conserve memory
# (in reality it leaks like a sieve if you enable it, so it's left on "False" for now)
RESTRICT_CACHE = False

# set to True to pretend to speed up GPU
# (in reality it just makes the graphics glitch, and then it slows to a crawl)
# NOTE: only applies to Pyglet routines!
CLEANUP_OLD_VERTICES = False

# set to False if you wish to use OpenGL 1.x features for stuff.
USE_SHADERS = True

# set to True if your GPU can handle it.
# this is the current shader, and it does specular reflection.
# some stuff will need to be backported to the "gouraud" shader.
USE_PHONG_SHADER = True

# draw methods:
# higher overrides lower
# if all are false, use pyglet.graphics.Batch (gets progressively worse)
USE_BUFFER_OBJECTS = False
USE_DISPLAY_LISTS = False

# please don't haphazardly tweak these, they're just to make the code nicer.
# [ nothing in particular here ]
# also some stuff that really should be defined but isn't.
GL_MAP_READ_BIT = 0x0001
GL_MAP_WRITE_BIT = 0x0002
# ok, you can screw with stuff again.

start_time = time.time()

# 0.0,0.0,0.15,0.0

if USE_PHONG_SHADER:
	SHADER_VERTEX_MAIN = """
//varying float fog;
varying vec4 phong_delta;
varying vec4 phong_normal;
varying vec4 phong_target;
void main()
{
	vec4 vec_base = gl_Vertex;
	vec4 vec_target = gl_ModelViewMatrix*vec_base;
	vec4 vec_normal = gl_ModelViewMatrix*-vec4(gl_Normal,0.0);
	gl_Position = gl_ProjectionMatrix*vec_target;
	vec4 vec_delta = vec_target;
	vec_delta[3] = 0.0;
	vec_delta = normalize(vec_delta);
	//float dist = length(vec3(vec_target));
	//fog = min(1.0, dist / *** PERCENT F! ***);
	phong_delta = vec_delta;
	phong_normal = vec_normal;
	phong_target = vec_target;
	//float diffuse = max(0.0, dot(vec_delta, vec_normal));
	//diffuse *= 6.0/max(6.0,length(vec3(vec_target)));
	//float diffuse = 1.0;
	gl_FrontColor = gl_Color;
	//gl_FrontColor = gl_Color * (0.97 * diffuse + 0.03);
	//gl_FrontColor = gl_Color * (0.97 * diffuse + 0.03) * (1.0-fog) + vec4(0.0,0.0,0.15,1.0) * fog;
}
	"""

	SHADER_FRAGMENT_MAIN = """
//varying float fog;
//varying float dist;
varying vec4 phong_delta;
varying vec4 phong_normal;
varying vec4 phong_target;

void main()
{
	// TODO: shift this to a uniform vector
	vec3 specular_vec = vec3(gl_ModelViewMatrix*-vec4(0.707, 0.707, 0.0, 0.0));
	
	//vec3 specular_vec_local = vec3(0.0, 0.0, -1.0);
	vec3 specular_vec_local = normalize(vec3(phong_delta));
	
	float spec_power_local = dot(vec3(0.0, 0.0, -1.0),vec3(phong_delta));
	spec_power_local = max(0.0, (spec_power_local-0.5)/(1.0-0.5));
	// acos(0.5) == 60 degrees
	
	vec3 col_base = normalize(vec3(gl_Color));
	float col_intens = length(vec3(gl_Color));
	//float diffuse = 1.0;
	float diffuse = max(0.0, spec_power_local*dot(specular_vec_local, vec3(phong_normal)));
	float dist = length(vec3(phong_target));
	diffuse *= 6.0/max(6.0,dist);
	diffuse = min(1.0, diffuse + max(0.0, dot(specular_vec, vec3(phong_normal))));
	float fog = max(0.0, min(1.0, dist * 2 / %f - 1.0));
	//float fog = 0.0;
	
	vec3 spec_vec = 2*dot(specular_vec,vec3(phong_normal))*vec3(phong_normal) - specular_vec;
	vec3 spec_vec_local = 2*dot(specular_vec_local,vec3(phong_normal))*vec3(phong_normal) - specular_vec_local;
	float specback = max(0.0, dot(spec_vec,normalize(vec3(phong_delta))));
	float specback2 = spec_power_local*max(0.0, dot(spec_vec_local,specular_vec_local));
	specback = pow(specback, 20);
	specback2 = pow(specback2, 20);
	//specback = 0.0;
	specback = min(1.0, specback+specback2);
	
	gl_FragColor = vec4(
			(col_base*col_intens*(0.97*diffuse + 0.03)*(1.0-specback)
				+ vec3(1.0,1.0,1.0)*specback
			)*(1.0-fog)
		,gl_Color[3])
		+ vec4(0.0,0.0,0.15,1.0) * fog;
	//gl_FragColor = vec4(col_base*col_intens*(1.0-fog),gl_Color[3]) + vec4(0.0,0.0,0.15,1.0) * fog;
	//gl_FragColor = vec4(col_base*col_intens,gl_Color[3]);
}
	""" % (DRAW_DIST)
else:
	SHADER_VERTEX_MAIN = """
varying float fog;

void main()
{
	vec4 vec_base = gl_Vertex;
	vec4 vec_target = gl_ModelViewMatrix*vec_base;
	vec4 vec_normal = gl_ModelViewMatrix*vec4(gl_Normal,0.0);
	gl_Position = gl_ProjectionMatrix*vec_target;
	vec4 vec_delta = vec_target;
	vec_delta[3] = 0.0;
	vec_delta = normalize(vec_delta);
	float dist = length(vec3(vec_target));
	fog = max(0.0, min(1.0, dist * 2 / %f - 1.0));
	float diffuse = max(0.0, dot(vec_delta, -vec_normal));
	diffuse *= 6.0/max(6.0,length(vec3(vec_target)));
	//float diffuse = 1.0;
	//gl_FrontColor = gl_Color;
	gl_FrontColor = gl_Color * (0.97 * diffuse + 0.03);
	//gl_FrontColor = gl_Color * (0.97 * diffuse + 0.03) * (1.0-fog) + vec4(0.0,0.0,0.15,1.0) * fog;
}
	""" % DRAW_DIST

	SHADER_FRAGMENT_MAIN = """
varying float fog;

void main()
{
	vec3 col_base = normalize(vec3(gl_Color));
	float col_intens = length(vec3(gl_Color));
	
	//gl_FragColor = vec4(col_base*col_intens*(0.97*diffuse + 0.03)*(1.0-fog),gl_Color[3]) + vec4(0.0,0.0,0.15,1.0) * fog;
	gl_FragColor = vec4(col_base*col_intens*(1.0-fog),gl_Color[3]) + vec4(0.0,0.0,0.15,1.0) * fog;
	//gl_FragColor = vec4(col_base*col_intens,gl_Color[3]);
}
	"""
class ClientArray:
	def __init__(self, typesize, veccount, atype):
		self.typesize = typesize
		self.veccount = veccount
		self.atype = atype
		self.size = 0
		self.buf = None
		self.index_end = 0
		self.tag_dict = {}
		self.unused_queue = []
		self.used_set = set()
		
		# allocate at least 8KB
		self.resize((8192+typesize-1)//(typesize))
	
	def get_pointer(self):
		return self.buf
	
	def write(self, idx, val_list):
		#print idx, val_list
		for val in val_list:
			self.buf[idx] = val
			idx += 1
	
	def allocate(self, tag):
		if tag in self.tag_dict:
			self.tag_dict[tag][0] += 1
			return self.tag_dict[tag][1], True
		elif self.unused_queue:
			idx = self.unused_queue.pop()
		else:
			idx = self.index_end
			self.index_end += 1
			self.require_size(self.index_end*self.veccount)
		
		self.tag_dict[tag] = [1,idx]
		self.used_set.add(idx)
		return idx, False
	
	def add(self, tag, val_list):
		idx, exists = self.allocate(tag)
		if not exists:
			self.write(idx*self.veccount, val_list)
		
		return idx
	
	def require_size(self, size):
		if self.size < size:
			# some of you Java programmers may find this a bit, um, familiar.
			# it's the same formula used in ArrayList.
			self.resize(max(size, (size * 3) // 2 + 1))
	
	def resize(self, size):
		print "RESIZE: %i -> %i" % (self.size, size)
		old_buf = self.buf
		
		self.buf = (self.atype * size)()
		
		if old_buf != None:
			for i in xrange(self.size):
				self.buf[i] = old_buf[i]
		
		self.size = size

class ArrayBufferHandle:
	def __init__(self):
		self.arr_vertex = ClientArray(4, 3, ctypes.c_float)
		self.arr_color = ClientArray(1, 4, ctypes.c_uint8)
		self.arr_index = ClientArray(2, 1, ctypes.c_uint16)
		self.vtx_list = []
	
	def add_vlist(self, ((ct,(r,g,b,_1,_2,_3,_4,_5,_6,_7,_8,_9)),
			(vt,(x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4))
				)):
		idx1 = self.add_vertex(r,g,b,x1,y1,z1)
		idx2 = self.add_vertex(r,g,b,x2,y2,z2)
		idx3 = self.add_vertex(r,g,b,x3,y3,z3)
		idx4 = self.add_vertex(r,g,b,x4,y4,z4)
		
		self.vtx_list = []
		tag_idx = (x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4,idx1,idx2,idx3,idx4)
		self.arr_index.add(tag_idx, (idx1,))
		self.arr_index.add(tag_idx, (idx2,))
		self.arr_index.add(tag_idx, (idx3,))
		self.arr_index.add(tag_idx, (idx4,))
		
		return (idx1,idx2,idx3,idx4,tag_idx)
	
	def add_vertex(self, r,g,b, x,y,z):
		tag = (r,g,b,x,y,z)
		idxv = self.arr_vertex.add(tag, (x,y,z))
		idxc = self.arr_color.add(tag, (r,g,b))
		
		assert idxv == idxc, "index mismatch: %i != %i" % (idxv, idxc)
		
		return idxv
	
	def draw(self):
		glEnableClientState(GL_VERTEX_ARRAY)
		glEnableClientState(GL_COLOR_ARRAY)
		glVertexPointer(3, GL_FLOAT, 0, self.arr_vertex.get_pointer())
		glColorPointer(3, GL_UNSIGNED_BYTE, 1, self.arr_color.get_pointer())
		
		arrptr = self.arr_index.get_pointer()
		arrsubptr = (ctypes.c_uint16 * 4)()
		for idx in self.arr_index.used_set:
			#print idx
			arrsubptr[:] = arrptr[idx:idx+4]
			glDrawElements(GL_QUADS, 4, GL_UNSIGNED_SHORT, arrsubptr)
		
		glDisableClientState(GL_COLOR_ARRAY)
		glDisableClientState(GL_VERTEX_ARRAY)
	
	def __str__(self):
		return "<buffer: size=[v=%i, c=%i, i=%i]>" % (
			self.arr_vertex.size,
			self.arr_color.size,
			self.arr_index.size,
				)

class Window:
	def __init__(self, width=800, height=600):
		self.width, self.height = width, height
		self.world = None
		self.win = pyglet.window.Window(
			caption="fireball",
			width=width,
			height=height,
			vsync=False)
		
		self.prep_gl()
		
		self.cam_x = 256.5
		self.cam_y = 256.5
		self.cam_z = 32.5
		self.cam_rz = 0.0
		self.cam_rx = 0.0
		self.cam_vx = 0.0
		self.cam_vy = 0.0
		
		self.old_xs = -1
		self.old_xe = -2
		self.old_ys = -1
		self.old_ye = -2
		
		self.lcx = -1
		self.lcy = -1
		
		self.cache_set = set()
		
		self.fps_display = pyglet.clock.ClockDisplay(color=(1.0,1.0,1.0,1.0))
		
		self.pillar_size = (DRAW_DIST + 15)*(DRAW_DIST + 15)
		self.pillar_index_insert = 0
		self.pillar_queue = np.tile(np.array((-1,-1), dtype=np.int16), (self.pillar_size,2)) # queue of pillars with vlists currently cached
		self.pillar_queue_set = set() # set to indicate if it's in or out
		
		self.win.set_mouse_visible(False)
		self.win.set_exclusive_mouse(True)
		
		self.win.push_handlers(self.on_draw)
		self.win.push_handlers(self.on_mouse_motion)
		self.win.push_handlers(self.on_key_press)
		self.win.push_handlers(self.on_key_release)
	
	def prep_shaders(self):
		self.shader_vtx = glCreateShader(GL_VERTEX_SHADER)
		tbuf = (ctypes.POINTER(ctypes.c_char) * 1)(ctypes.create_string_buffer(SHADER_VERTEX_MAIN))
		glShaderSource(self.shader_vtx, 1, tbuf, None)
		glCompileShader(self.shader_vtx)
		
		self.shader_frag = glCreateShader(GL_FRAGMENT_SHADER)
		tbuf = (ctypes.POINTER(ctypes.c_char) * 1)(ctypes.create_string_buffer(SHADER_FRAGMENT_MAIN))
		glShaderSource(self.shader_frag, 1, tbuf, None)
		glCompileShader(self.shader_frag)
		
		self.shader_program = glCreateProgram()
		glAttachShader(self.shader_program, self.shader_vtx)
		glAttachShader(self.shader_program, self.shader_frag)
		glLinkProgram(self.shader_program)
	
	def prep_gl(self):
		# set some flags
		glEnable(GL_DEPTH_TEST)
		glEnable(GL_CULL_FACE)
		glDepthRange(0.0, 1.0)
		glDepthFunc(GL_LEQUAL)
		
		# set up fog
		self.draw_dist_fog = DRAW_DIST / sqrt(1.0 + (float(self.width)/self.height)**2)
		cc = [0.0,0.0,0.15,0.0]
		glClearColor(*cc)
		if not USE_SHADERS:
			glFogfv(GL_FOG_COLOR, (ctypes.c_float * 4)(*cc))
			glFogi(GL_FOG_MODE, GL_LINEAR)
			glFogf(GL_FOG_DENSITY, 2.0)
			glFogf(GL_FOG_START, float(self.draw_dist_fog)/2.0)
			glFogf(GL_FOG_END, float(self.draw_dist_fog))
		
		# set up shader if necessary
		if USE_SHADERS:
			self.prep_shaders()
		
		# set up desired rendering system
		if USE_BUFFER_OBJECTS:
			self.gl_buffer_list = (ctypes.c_uint32 * 2)(0, 0)
			glGenBuffers(2, self.gl_buffer_list)
			self.buffer_1 = ArrayBufferHandle()
			print "buffer objects:", self.buffer_1
		elif USE_DISPLAY_LISTS:
			self.dlist_set = set()
			self.dlist_nestings = {}
			self.dlist_nesting_list = {}
			self.dlist_nesting_list_update = set()
			self.dlist_main = glGenLists(1)
		else:
			# rely on Pyglet to leak lots of memory
			self.batch = pyglet.graphics.Batch()
			self.batch_queue = []
		
		# build some objects
		self.draw_cube()
		pass
		
	def draw_cube(self):
		# GL: -xyz +xyz
		# coords: -xzy +xzy
		ql = [
			('v3f',(0.0, 0.0, 0.0,
				0.0, 0.0, 1.0,
				0.0, 1.0, 1.0,
				0.0, 1.0, 0.0)),
			('v3f',(0.0, 0.0, 0.0,
				1.0, 0.0, 0.0,
				1.0, 0.0, 1.0,
				0.0, 0.0, 1.0)),
			('v3f',(0.0, 0.0, 0.0,
				0.0, 1.0, 0.0,
				1.0, 1.0, 0.0,
				1.0, 0.0, 0.0)),
			('v3f',(1.0, 0.0, 0.0,
				1.0, 1.0, 0.0,
				1.0, 1.0, 1.0,
				1.0, 0.0, 1.0)),
			('v3f',(0.0, 1.0, 0.0,
				0.0, 1.0, 1.0,
				1.0, 1.0, 1.0,
				1.0, 1.0, 0.0)),
			('v3f',(0.0, 0.0, 1.0,
				1.0, 0.0, 1.0,
				1.0, 1.0, 1.0,
				0.0, 1.0, 1.0)),
		]
		
		self.vlist_cube = ql
		#self.obj_cube = pyglet.graphics.Batch()
		#for q in ql:
		#	self.obj_cube.add(4, GL_QUADS, None, q)
	
	def set_game(self, game):
		self.game = game
	
	def set_world(self, world):
		self.world = world
		self.cam_x = self.world.lx//2 + 0.5
		self.cam_y = self.world.ly//2 + 0.5
		self.cam_z = 32.5
	
	def set_controller(self, controller):
		self.controller = controller
	
	def mainloop(self):
		print "Starting!"
		if USE_EXPLICIT_LOOP:
			pyglet.clock.schedule_interval(self.cleanup_some_crap, 4.0)
			baseclock = pyglet.clock.Clock()
			while not self.win.has_exit:
				self.win.dispatch_events()
				self.win.dispatch_event('on_draw')
				#self.on_draw()
				self.update_ents(baseclock.tick())
				pyglet.clock.tick()
				self.win.flip()
		else:
			pyglet.clock.schedule_interval(self.update_ents, PHYSICS_SPF)
			pyglet.clock.schedule_interval(self.cleanup_some_crap, 4.0)
			pyglet.app.run()
	
	def update_ents(self, dt):
		if self.game != None:
			self.game.update(float(dt))
	
	def cleanup_some_crap(self, dt):
		#print "GC:",gc.collect()
		if CLEANUP_OLD_VERTICES:
			ctr = min(len(self.batch_queue),max(100,min(3000,len(self.batch_queue)*2//3+1)))
			if ctr != 0:
				old_vl = self.batch_queue[:ctr]
				print "vlist remove %i/%i" % (ctr,len(self.batch_queue))
				for vl in old_vl:
					vl.delete()
				self.batch_queue = self.batch_queue[ctr:]
	
	def on_mouse_motion(self, x, y, dx, dy):
		if self.controller != None:
			self.controller.on_mouse_motion(x,y,dx,dy)
	
	def on_key_press(self, key, mod):
		if self.controller != None:
			self.controller.on_key_press(key,mod)
	
	def on_key_release(self, key, mod):
		if self.controller != None:
			self.controller.on_key_release(key,mod)
	
	def on_draw(self):
		#print "poo"
		# clear screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		
		if self.world and self.world.ready:
			self.draw_world()
		else:
			pass
		
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
		glClear(GL_DEPTH_BUFFER_BIT)
		glTranslatef(100.0,-160.0,-200.0)
		if USE_SHADERS:
			glUseProgram(0)
		else:
			glDisable(GL_FOG)
		self.fps_display.draw()
		
		self.win.flip()
	
	def draw_world(self):
		#print "draw %.5f %.2f" % (time.time() - start_time, pyglet.clock.get_fps())
		
		# enable fog
		if USE_SHADERS:
			glUseProgram(self.shader_program)
		else:
			glEnable(GL_FOG)
		
		# set projection matrix
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		scale = float(self.width)/self.height
		gluPerspective(90.0, scale, 0.01, 200.0)
		
		# prep camera
		self.cam_x, self.cam_y, self.cam_z, self.cam_rz, self.cam_rx = self.controller.set_camera()
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
		glRotatef(self.cam_rx, 1.0, 0.0, 0.0)
		glRotatef(self.cam_rz, 0.0, 1.0, 0.0)
		glTranslatef(-self.cam_x, -self.cam_z, -self.cam_y)
		
		# draw stuff
		cx,cy,cz = int(self.cam_x+0.5), int(self.cam_y+0.5), int(self.cam_z+0.5)
		
		if cx < DRAW_DIST:
			cx = DRAW_DIST
		if cx > self.world.lx-1-DRAW_DIST:
			cx = self.world.lx-1-DRAW_DIST
		
		if cy < DRAW_DIST:
			cy = DRAW_DIST
		if cy > self.world.ly-1-DRAW_DIST:
			cy = self.world.ly-1-DRAW_DIST
		
		
		xs,xe = (max(0,min(self.world.lx-1, cx+q)) for q in (-DRAW_DIST,DRAW_DIST))
		ys,ye = (max(0,min(self.world.ly-1, cy+q)) for q in (-DRAW_DIST,DRAW_DIST))
		#zs,ze = (max(0,min(self.world.lz-1, cz+q)) for q in (-8,8))
		
		# check if the old and new ranges intersect at all
		if self.old_xs >= xe or self.old_xe <= xs or self.old_ys >= ye or self.old_ye <= ys:
			# completely unintersecting
			self.remove_pillars(self.old_xs, self.old_xe, self.old_ys, self.old_ye, False)
			self.add_pillars(xs, xe, ys, ye)
		elif self.old_xs == xs and self.old_xe == xe and self.old_ys == ys and self.old_ye == ye:
			# completely overlapping
			pass 
		else:
			# partial overlaps
			#
			# +-----+
			# |     |
			# |==+--+--+
			# |  |  |  |
			# +--+--+==|
			#    |     |
			#    +-----+
			#
			if xs < self.old_xs:
				self.remove_pillars(xe, self.old_xe, max(ys,self.old_ys), min(ye,self.old_ye), False)
				self.add_pillars(xs, self.old_xs, max(ys,self.old_ys), min(ye,self.old_ye))
			elif xs > self.old_xs:
				self.remove_pillars(self.old_xs, xs, max(ys,self.old_ys), min(ye,self.old_ye), False)
				self.add_pillars(self.old_xe, xe, max(ys,self.old_ys), min(ye,self.old_ye))
			
			if ys < self.old_ys:
				self.remove_pillars(self.old_xs, self.old_xe, ye, self.old_ye, False)
				self.add_pillars(xs, xe, ys, self.old_ys)
			elif ys > self.old_ys:
				self.remove_pillars(self.old_xs, self.old_xe, self.old_ys, ys, False)
				self.add_pillars(xs, xe, self.old_ye, ye)
			
			#self.remove_pillars(self.old_xs, self.old_xe, self.old_ys, self.old_ye, True, xs,xe,ys,ye)
			#self.add_pillars(xs, xe, ys, ye)
		
		pmap = self.world.get_pillar_array()
		
		# update queued pillar caching
		if len(self.cache_set) > 0:
			ccount = min(len(self.cache_set), max(DRAW_DIST*3//2,len(self.cache_set)//4))
			print "cache %i/%i" % (ccount, len(self.cache_set))
			for i in xrange(ccount):
				x,y = self.cache_set.pop()
				pmap[x][y][0] = self.cache_pillar(x,y)
				self.add_pillar_vlist(x,y)
		
		# save new range
		self.old_xs, self.old_xe = xs, xe
		self.old_ys, self.old_ye = ys, ye
		
		# draw it
		if USE_BUFFER_OBJECTS:
			self.buffer_1.draw()
		elif USE_DISPLAY_LISTS:
			if self.dlist_nesting_list_update:
				for y in self.dlist_nesting_list_update:
					if y not in self.dlist_nesting_list:
						self.dlist_nesting_list[y] = None
					
					list_id = glGenLists(1) if self.dlist_nesting_list[y] == None else self.dlist_nesting_list[y]
					glNewList(list_id, GL_COMPILE)
					for sublist_id in self.dlist_nestings[y]:
						glCallList(sublist_id)
					glEndList()
					self.dlist_nesting_list[y] = list_id
				
				self.dlist_nesting_list_update.clear()
				
				glNewList(self.dlist_main, GL_COMPILE_AND_EXECUTE)
				for y in self.dlist_nesting_list:
					list_id = self.dlist_nesting_list[y]
					glCallList(list_id)
				glEndList()
			else:
				glCallList(self.dlist_main)
		else:
			self.batch.draw()
		
	def add_pillars(self, xs, xe, ys, ye):
		pmap = self.world.get_pillar_array()
		for x in xrange(xs,xe):
			for y in xrange(ys,ye):
				pm_list, pm_handle, pm_def = pmap[x][y]
				if not pm_list:
					self.precache_pillar(x,y)
					continue
				if not pm_handle:
					pm_handle = self.add_pillar_vlist(x, y)
	
	def add_pillar_vlist(self, x, y):
		pmap = self.world.get_pillar_array()
		r = None
		if USE_BUFFER_OBJECTS:
			r = [
				self.buffer_1.add_vlist(vl)
				for vl in pmap[x][y][0]] # FIXME / TODO: actually return SOMETHING
		elif USE_DISPLAY_LISTS:
			r = [
				self.dlist_allocate(x, y, vl)
				for vl in pmap[x][y][0]]
		else:
			r = [
				self.batch_add(vl)
				for vl in pmap[x][y][0]]
		pmap[x][y][1] = r
		
		return r
	
	def dlist_allocate(self, x, y, vl):
		list_id = glGenLists(1)
		
		glNewList(list_id, GL_COMPILE)
		pyglet.graphics.draw(4, GL_QUADS, *vl)
		glEndList()
		
		self.dlist_set.add(list_id)
		if y not in self.dlist_nestings:
			self.dlist_nestings[y] = set()
		
		self.dlist_nestings[y].add(list_id)
		self.dlist_nesting_list_update.add(y)
		
		return list_id
	
	def remove_pillars(self, xs, xe, ys, ye, doischeck, ixs=None,ixe=None,iys=None,iye=None):
		wmap = self.world.get_block_array()
		pmap = self.world.get_pillar_array()
		for x in xrange(xs,xe):
			for y in xrange(ys,ye):
				if doischeck and x >= ixs and x < ixe and y >= iys and y < iye:
					continue
				
				if (x,y) in self.cache_set:
					self.cache_set.remove((x,y))
				
				if pmap[x][y][1] != None:
					if USE_BUFFER_OBJECTS:
						pass
					elif USE_DISPLAY_LISTS:
						for vl in pmap[x][y][1]:
							glDeleteLists(vl, 1)
							if vl in self.dlist_set:
								self.dlist_set.remove(vl)
							if y in self.dlist_nestings and vl in self.dlist_nestings[y]:
								self.dlist_nestings[y].remove(vl)
								self.dlist_nesting_list_update.add(y)
							
					else:
						for vl in pmap[x][y][1]:
							self.batch_queue.append(vl)
					
					pmap[x][y][1] = None
	
	def batch_add(self, vl):
		if self.batch_queue:
			x = self.batch_queue.pop()
			for i in xrange(12):
				x.colors[i] = vl[0][1][i]
				x.vertices[i] = vl[1][1][i]
				x.normals[i] = vl[2][1][i]
			return x
		else:
			x = self.batch.add(4, GL_QUADS, None, *((s+"/dynamic",t) for (s,t) in vl))
			return x
	
	def precache_pillar(self, x, y):
		self.cache_set.add((x,y))
	
	def cache_pillar(self, x, y):
		#print "caching",x,y
		wmap = self.world.get_block_array()
		pmap = self.world.get_pillar_array()
		pm_list, pm_handle, pm_def = pmap[x][y]
		
		vl = []
		
		for (zs,ze) in pm_def:
			assert zs < ze, "empty/reverse runs should not exist"
			
			# add top
			if self.world.test_if_air(x,y,ze):
				z = ze-1
				r,g,b,t = wmap[x][y][ze-1]
				assert t&world.BTYPE_MASK_TYPE != 0, "pillar array should not contain runs of air!"
				r,g,b = int(r),int(g),int(b)
				cvalz = ('c3B',(r,g,b)*4)
				t,ft = self.vlist_cube[4]
				x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4 = ft
				ft = (
					x1+x,y1+z,z1+y,
					x2+x,y2+z,z2+y,
					x3+x,y3+z,z3+y,
					x4+x,y4+z,z4+y,
				)
				vl.append((cvalz,(t,ft),('n3f',(0.0,1.0,0.0)*4)))
				#vl.append((cvalz,(t,ft),('n3f',
				#	(
				#		-0.5773,0.5773,-0.5773,
				#		-0.5773,0.5773,0.5773,
				#		0.5773,0.5773,0.5773,
				#		0.5773,0.5773,-0.5773,
				#	),
				#		)))
			
			# add bottom
			if self.world.test_if_air(x,y,zs-1):
				z = zs
				r,g,b,t = wmap[x][y][zs]
				assert t&world.BTYPE_MASK_TYPE != 0, "pillar array should not contain runs of air!"
				r,g,b = int(r),int(g),int(b)
				cvalz = ('c3B',(r,g,b)*4)
				t,ft = self.vlist_cube[1]
				x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4 = ft
				ft = (
					x1+x,y1+z,z1+y,
					x2+x,y2+z,z2+y,
					x3+x,y3+z,z3+y,
					x4+x,y4+z,z4+y,
				)
				vl.append((cvalz,(t,ft),('n3f',(0.0,-1.0,0.0)*4)))
			
			vlist_cube_sides = self.vlist_cube[0:][:1] + self.vlist_cube[2:][:2] + self.vlist_cube[5:][:1]
			
			for z in xrange(zs,ze):
				# get block
				r,g,b,t = wmap[x][y][z]
				assert t&world.BTYPE_MASK_TYPE != 0, "pillar array should not contain runs of air!"
				
				# add our tupletastic colour tuple
				r,g,b = int(r),int(g),int(b)
				
				if USE_SHADERS:
					# shader will handle diffuse lighting
					cvalx = cvaly = ('c3B',(r,g,b)*4)
				else:
					cvalx = ('c3B',(r//2,g//2,b//2)*4)
					cvaly = ('c3B',(r*3//4,g*3//4,b*3//4)*4)
				
				# work out each face's visibility
				for (u,v,w,cv),(t,ft) in zip([
							(-1,0,0,cvalx),
							(0,-1,0,cvaly),
							(1,0,0,cvalx),
							(0,1,0,cvaly),
						],vlist_cube_sides):
					
					if self.world.test_if_air(x+u,y+v,z+w):
						x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4 = ft
						ft = (
							x1+x,y1+z,z1+y,
							x2+x,y2+z,z2+y,
							x3+x,y3+z,z3+y,
							x4+x,y4+z,z4+y,
						)
						#print ft
						vl.append((cv,(t,ft),('n3f',(float(u),float(w),float(v))*4)))
		
		pm_list = vl
		
		pmap[x][y][0] = pm_list
		
		# ensure the cache doesn't get TOO large
		if RESTRICT_CACHE and x*32768+y not in self.pillar_queue_set:
			rv = self.pillar_queue[self.pillar_index_insert]
			if rv[0] != -1:
				rx,ry = rv[0],rv[1]
				self.pillar_queue_set.remove(rx*32768+ry)
				pmap[x][y][0] = None
				pmap[x][y][1] = None
				
			rv[0] = x
			rv[1] = y
			self.pillar_index_insert += 1
			while self.pillar_index_insert >= len(self.pillar_queue):
				self.pillar_index_insert -= len(self.pillar_queue)
			
			self.pillar_queue_set.add(x*32768+y)
		
		return pm_list
	
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
