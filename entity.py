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

from math import *

import numpy as np

import pyglet

import helpers

MOUSE_SENS_X = 0.3
MOUSE_SENS_Y = 0.3
PLAYER_SPEED = 3.0*2.0
OBJECT_GRAVITY = 9.8*2.0
PLAYER_FRICTION = 0.02
PLAYER_JUMP_HEIGHT = 10.0
COLLISION_TOLERANCE = 0.2

KEY_MOVE_FORWARD_BIT = 0x0001
KEY_MOVE_BACKWARD_BIT = 0x0002
KEY_MOVE_LEFT_BIT = 0x0004
KEY_MOVE_RIGHT_BIT = 0x0008
KEY_JUMP_BIT = 0x0010
KEY_CROUCH_BIT = 0x0020
KEY_CREEP_BIT = 0x0040
KEY_ZOOM_BIT = 0x0080

class AbstractEntity(helpers.ArgGenerator):
	ARGS = []
	
	def set_game(self, idx, game):
		self.idx = idx
		self.game = game

class PositionedEntity(AbstractEntity):
	ARGS = AbstractEntity.ARGS + ["origin","velocity","orient_x","orient_z"]

class PhysicsEntity(PositionedEntity):
	ARGS = PositionedEntity.ARGS + []
	grounded = False
	walkable = False
	
	# i had to use floor,
	# otherwise the player would bounce like mad when it was in the water
	def trace_vector(self, ox,oy,oz, nx,ny,nz, walkable = False):
		#walkable = False
		
		# prep values
		dx, dy, dz = (n-o for (o,n) in zip((ox,oy,oz),(nx,ny,nz))) # delta
		
		(x1,y1,z1), (x2,y2,z2) = self.BBOX
		height = floor(abs(z2-z1)-0.001)+1
		
		x3, y3, z3 = (v1 if d < 0.0 else v2 for (v1,v2,d) in zip(self.BBOX[0], self.BBOX[1], (dx, dy, dz)))
		x4, y4, z4 = (v2-v1 if d < 0.0 else v1-v2 for (v1,v2,d) in zip(self.BBOX[0], self.BBOX[1], (dx, dy, dz)))
		
		z5 = (0.0 if dz < 0.0 else z4)
		
		ox += x3
		oy += y3
		oz += z3
		
		nx += x3
		ny += y3
		nz += z3
		
		sx, sy, sz = (v%1.0 if d < 0.0 else 1.0-(v%1.0) for v,d in zip((ox,oy,oz),(dx,dy,dz))) # sub
		gx, gy, gz = (-1 if d < 0.0 else 1 for d in (dx, dy, dz)) # direction ("go")
		wx, wy, wz = (0.001 if d < 0.0 else 0.999 for d in (dx, dy, dz)) # cell offset when hitting box
		vx, vy, vz = (max(0.00001,abs(d)) for d in (dx, dy, dz)) # abs velocity
		cx, cy, cz = (int(floor(v)) for v in (ox, oy, oz)) # cell
		dcx, dcy, dcz = (abs(int(floor(v))-c) for c,v in zip((cx,cy,cz),(nx,ny,nz))) # cell delta / count
		
		walkable = walkable and dz < 0.0
		
		def sfix(sx,sy,sz):
			return tuple(v if d < 0.0 else 1.0-v for (v,d) in zip((sx,sy,sz),(dx,dy,dz)))
		
		# flags to indicate if we've screwed with a value
		keep_x = True
		keep_y = True
		keep_z = True
		
		dc = dcx+dcy+dcz
		
		for i in xrange(dc):
			# get our lovely factoriffic stuff
			calc_x = sx/vx
			calc_y = sy/vy
			calc_z = sz/vz
			
			take_x = calc_x < calc_y and calc_x < calc_z
			take_y = (not take_x) and calc_y < calc_z
			take_z = (not take_x) and (not take_y)
			
			if take_x:
				# X trace
				t = sx/vx
				sy -= t*vy
				sz -= t*vz
				
				if keep_x:
					cx += gx
				sx = 1.0
			elif take_y:
				# Y trace
				t = sy/vy
				sx -= t*vx
				sz -= t*vz
				
				if keep_y:
					cy += gy
				sy = 1.0
			else:
				# Z trace
				t = sz/vz
				sx -= t*vx
				sy -= t*vy
				
				if keep_z:
					cz += gz
				sz = 1.0
			
			# cell check!
			
			ax,ay,az = sfix(sx,sy,sz) # add this to cx,cy,cz
			ncx,ncy,ncz = cx+ax,cy+ay,cz+az
			if not keep_x:
				ncx = nx
			if not keep_y:
				ncy = ny
			if not keep_z:
				ncz = nz
			
			if take_x:
				floor_check = not self.game.world.solid_check_box(
					cx+0.5-gx,ncy,ncz+1,
					cx+0.5,ncy+y4,ncz+z4+1
						)
				checked_out_as_solid = self.game.world.solid_check_box(
					cx+0.5-gx,ncy,ncz,
					cx+0.5,ncy+y4,ncz+z4
						)
			elif take_y:
				floor_check = not self.game.world.solid_check_box(
					ncx,cy+0.5-gy,ncz+1,
					ncx+x4,cy+0.5,ncz+z4+1
						)
				checked_out_as_solid = self.game.world.solid_check_box(
					ncx,cy+0.5-gy,ncz,
					ncx+x4,cy+0.5,ncz+z4
						)
			else:
				checked_out_as_solid = self.game.world.solid_check_box(
					ncx,ncy,cz+0.5-gz,
					ncx+x4,ncy+y4,cz+0.5
						)
			
			#if self.game.world.test_if_solid(cx,cy,cz):
			if checked_out_as_solid:
				if take_x:
					if walkable and keep_x and floor_check:
						cz += 1
						onz = nz
						nz = cz+0.001
						self.antijerk_stairs += onz-nz
						keep_x = False
					else:
						cx -= gx
						#sx = 0.1
						if keep_x:
							nx = cx+wx
							self.velocity[0] *= -0.1
							keep_x = False
				elif take_y:
					if walkable and keep_y and floor_check:
						cz += 1
						onz = nz
						nz = cz+0.001
						self.antijerk_stairs += onz-nz
						keep_z = False
					else:
						cy -= gy
						#sy = 0.1
						if keep_y:
							ny = cy+wy
							self.velocity[1] *= -0.1
							keep_y = False
				elif take_z:
					cz -= gz
					#sz = 0.1
					if keep_z:
						nz = cz+wz
						
						if gz < 0:
							self.grounded = True
						self.velocity[2] *= -0.1
						keep_z = False
		
		return nx-x3, ny-y3, nz-z3
	
	def update(self, dt):
		# get new position
		nvec = tuple(self.origin[i] + self.velocity[i]*dt for i in xrange(3))
		
		(x1, y1, z1), (x2, y2, z2) = self.BBOX
		
		ox, oy, oz = self.origin
		nx, ny, nz = nvec
		
		# trace each corner
		
		#for vbase in self.BVEC:
		#	vx, vy, vz, walkable = vbase
		#	tnx, tny, tnz = self.trace_vector(ox+vx, oy+vy, oz+vz, nx+vx, ny+vy, nz+vz, walkable)
		#	nx, ny, nz = (v-vo for (v,vo) in zip((tnx,tny,tnz),(vx,vy,vz)))
		
		nx, ny, nz = self.trace_vector(ox, oy, oz, nx, ny, nz, self.walkable)
		
		for i,vt in zip(xrange(3), (nx, ny, nz)):
			self.origin[i] = vt

class PlayerEntity(PhysicsEntity):
	ARGS = PhysicsEntity.ARGS + ["name","keys"]
	BBOX_STAND = ((-0.4, -0.4, -2.4),(0.4, 0.4, 0.4))
	BBOX_CROUCH = ((-0.4, -0.4, -1.4),(0.4, 0.4, 0.4))
	
	BBOX = BBOX_STAND
	
	def set_game(self, idx, game):
		self.idx = idx
		self.game = game
		
		self.target_velocity = [0.0, 0.0, 0.0]
		self.cam_vx = self.cam_vy = 0.0
		self.antijerk_stairs = 0.0
		self.crouching = False
		self.walkable = True
		
		if game != None:
			# init
			if self.origin == None:
				x = self.game.world.lx//2 + 0.5
				y = self.game.world.ly//2 + 0.5
				z = self.game.world.lz + 0.5
				self.origin = [x,y,z]
			
			if self.orient_x == None:
				self.orient_x = 0.0
			if self.orient_z == None:
				self.orient_z = 0.0
			
			if self.velocity == None:
				self.velocity = [0.0, 0.0, 0.0]
			
			if self.keys == None:
				self.keys = 0
			
			if self.name == None:
				self.name = "Griefer" + repr(self.idx)
		else:
			# destroy
			pass
	
	def set_camera(self):
		x,y,z = self.origin
		return x,y,z+self.antijerk_stairs,self.orient_z,self.orient_x
	
	def update(self, dt):
		#print dt
		cam_rmatrix = self.get_cam_matrix_noxrot()
		
		self.cam_vx = 0.0
		self.cam_vy = 0.0
		
		# fix antijerk
		self.antijerk_stairs *= exp(-10.0*dt)
		
		# deal with key changes
		if (self.keys & KEY_JUMP_BIT) and self.grounded and not self.crouching:
			self.velocity[2] = PLAYER_JUMP_HEIGHT
			self.grounded = False
		
		if (self.keys & KEY_MOVE_LEFT_BIT):
			if not (self.keys & KEY_MOVE_RIGHT_BIT):
				self.cam_vx = -1.0
		elif (self.keys & KEY_MOVE_RIGHT_BIT):
			self.cam_vx = 1.0
		
		if (self.keys & KEY_MOVE_BACKWARD_BIT):
			if not (self.keys & KEY_MOVE_FORWARD_BIT):
				self.cam_vy = -1.0
		elif (self.keys & KEY_MOVE_FORWARD_BIT):
			self.cam_vy = 1.0
		
		bvx = self.cam_vx*PLAYER_SPEED
		bvy = -self.cam_vy*PLAYER_SPEED
		
		if bool(self.keys & KEY_CROUCH_BIT) != self.crouching:
			if self.crouching:
				# uncrouch check
				(x1,y1,z1),(x2,y2,z2) = self.BBOX_STAND
				x,y,z = self.origin
				
				if not self.game.world.solid_check_box(x1+x,y1+y,z1+z+2,x2+x,y2+y,z2+z+0.1+1):
					self.origin[2] += 1.0
					self.BBOX = self.BBOX_STAND
					self.antijerk_stairs -= 1.0
					self.crouching = False
					self.walkable = True 
			else:
				# crouch - no check needed
				self.origin[2] -= 1.0
				self.BBOX = self.BBOX_CROUCH
				self.antijerk_stairs += 1.0
				self.crouching = True
				self.walkable = False
		
		if (self.keys & KEY_CREEP_BIT) or self.crouching:
			bvx *= 0.5
			bvy *= 0.5
		
		q = (np.asmatrix([bvx,bvy,0.0])*cam_rmatrix)
		#for i in xrange(3):
		#	self.velocity[i] *= (1.0-PLAYER_FRICTION*dt)
		
		self.target_velocity[0] = q[0,0]
		self.target_velocity[1] = q[0,1]
		self.target_velocity[2] = q[0,2]
		
		for i in [0,1]: # don't do this with Z.
		#for i in [0,1,2]: # ok, maybe as a temp measure
			# TODO: get the math behind this right
			self.velocity[i] += (self.target_velocity[i] - self.velocity[i])*(1.0 - exp(-dt*5.0))
		
		self.velocity[2] -= OBJECT_GRAVITY*dt
		
		PhysicsEntity.update(self, dt)
	
	
	def get_cam_matrix_noxrot(self):
		srz,crz = sin(self.orient_z*pi/180.0),cos(self.orient_z*pi/180.0)
		
		cam_rmatrix = np.asmatrix(np.identity(3))
		
		cam_rmatrix *= np.asmatrix([
			[crz,srz,0.0],
			[-srz,crz,0.0],
			[0.0,0.0,1.0],
		])
		
		return cam_rmatrix
	
	def get_cam_matrix(self):
		srx,crx = sin(self.orient_x*pi/180.0),cos(self.orient_x*pi/180.0)
		srz,crz = sin(self.orient_z*pi/180.0),cos(self.orient_z*pi/180.0)
		
		cam_rmatrix = np.asmatrix(np.identity(3))
		
		cam_rmatrix *= np.asmatrix([
			[1.0,0.0,0.0],
			[0.0,crx,srx],
			[0.0,srx,-crx],
		])
		
		cam_rmatrix *= np.asmatrix([
			[crz,srz,0.0],
			[-srz,crz,0.0],
			[0.0,0.0,1.0],
		])
		
		return cam_rmatrix
	
	def on_mouse_motion(self, x, y, dx, dy):
		self.orient_z += dx*MOUSE_SENS_X
		self.orient_x -= dy*MOUSE_SENS_Y
	
	def on_key_press(self, key, mod):
		if key == pyglet.window.key.W:
			self.keys |= KEY_MOVE_FORWARD_BIT
		elif key == pyglet.window.key.S:
			self.keys |= KEY_MOVE_BACKWARD_BIT
		elif key == pyglet.window.key.A:
			self.keys |= KEY_MOVE_LEFT_BIT
		elif key == pyglet.window.key.D:
			self.keys |= KEY_MOVE_RIGHT_BIT
		elif key == pyglet.window.key.SPACE:
			self.keys |= KEY_JUMP_BIT
		elif key == pyglet.window.key.LCTRL:
			self.keys |= KEY_CROUCH_BIT
		elif key == pyglet.window.key.LSHIFT:
			self.keys |= KEY_CREEP_BIT
	
	def on_key_release(self, key, mod):
		if key == pyglet.window.key.W:
			self.keys &= ~KEY_MOVE_FORWARD_BIT
		elif key == pyglet.window.key.S:
			self.keys &= ~KEY_MOVE_BACKWARD_BIT
		elif key == pyglet.window.key.A:
			self.keys &= ~KEY_MOVE_LEFT_BIT
		elif key == pyglet.window.key.D:
			self.keys &= ~KEY_MOVE_RIGHT_BIT
		elif key == pyglet.window.key.SPACE:
			self.keys &= ~KEY_JUMP_BIT
		elif key == pyglet.window.key.LCTRL:
			self.keys &= ~KEY_CROUCH_BIT
		elif key == pyglet.window.key.LSHIFT:
			self.keys &= ~KEY_CREEP_BIT
