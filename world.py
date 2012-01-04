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

import random
from math import *

import numpy as np

import helpers

def to_rgb_t(r,g,b,t):
	#return np.int32(r|(g<<8)|(b<<16)|(t<<24))
	return np.array((r,g,b,t),dtype=np.uint8)

def to_rgb_tx(r,g,b,t):
	return tuple(np.uint8(v) for v in (r,g,b,t))

BTYPE_FLAG_UNBREAKABLE = 0x40

BTYPE_MASK_TYPE = 0x3C

BTYPE_AIR = 0
BTYPE_SOLID = 4
BTYPE_WATER = 8

# TODO: update this for int8*4 (vs int32*1)
def gen_flat(solheight):
	def _gen(self, lx, ly, lz):
		print "MAPGEN: Creating types"
		b_water = to_rgb_tx(128,128,240,BTYPE_WATER|BTYPE_FLAG_UNBREAKABLE)
		b_solid = to_rgb_tx(128,240,128,BTYPE_SOLID|BTYPE_FLAG_UNBREAKABLE)
		b_air = to_rgb_tx(0,0,0,BTYPE_AIR)
		
		print "MAPGEN: Creating template"
		template = (
			  [b_water]
			+ [b_solid]*(solheight-1)
			+ [b_air]*(lz-solheight)
		)
		
		print "MAPGEN: Building array from template"
		self.g = np.tile(np.array(template, dtype=np.uint8), reps=(lx,ly,1,1))
		
		# current bottleneck.
		# i'm a bit fearful that speeding this up by using np.fill will break stuff.
		print "MAPGEN: Pillarating"
		self.p = np.ndarray(shape=(self.lx,self.ly,3), dtype=np.object_)
		for x in xrange(lx):
			for y in xrange(ly):
				self.p[x][y][0] = None
				self.p[x][y][1] = None
				self.p[x][y][2] = [(solheight-1,solheight)]
				
		print "MAPGEN: Done!"
	
	return _gen

def gen_white(density):
	def _gen(self, lx, ly, lz):
		print "MAPGEN: Building blank array"
		self.g = np.zeros(
			shape=(lx,ly,lz,4),
			dtype=np.int8
		)
		
		# TODO: use random.expovariate or something along those lines (if possible)
		print "MAPGEN: Filling map with noise"
		for x in xrange(lx):
			for y in xrange(ly):
				for z in xrange(1,lz):
					if random.random() < density:
						print x,y,z
						self.g[x][y][z] = to_rgb_t(
							random.randint(0,255),
							random.randint(0,255),
							random.randint(0,255),
							BTYPE_SOLID
						)
		
		b_water = to_rgb_t(128,128,240,BTYPE_WATER|BTYPE_FLAG_UNBREAKABLE)
		
		print "MAPGEN: Fixing layer 0 (water)"
		for x in xrange(lx):
			for y in xrange(ly):
				self.g[x][y][0] = b_water
		
		print "MAPGEN: Protecting layer 1 (solid)"
		flg = BTYPE_FLAG_UNBREAKABLE
		for x in xrange(lx):
			for y in xrange(ly):
				self.g[x][y][1][3] |= flg
		
		print "MAPGEN: Done!"
	return _gen

# NOTE: This should be fast enough to load a full AoS map now.
def gen_aos_vxl(fname, offs=(0,0), size=(512,512)):
	sizex, sizey = size
	offsx, offsy = offs
	def _gen(self, lx, ly, lz):
		#assert lx == 512 and ly == 512 and lz == 64, "Dimensions incorrect for AoS map (should be 512x512x64)"
		assert lz == 64, "Height incorrect for AoS map (should be 64)"
		assert lx <= sizex and ly <= sizex, "Cannot upscale an AoS map (max 512x512 x,y)"
		
		b_air = to_rgb_tx(0,0,0,BTYPE_AIR)
		b_poo = to_rgb_tx(170//2,85//2,0,BTYPE_SOLID)
		b_poo_fail = to_rgb_tx(255,0,0,BTYPE_SOLID)
		print "MAPGEN: Preparing block array"
		#self.g = np.zeros(
		#	shape=(lx,ly,lz,4),
		#	dtype=np.int8
		#)
		
		self.g = np.tile(b_poo,(lx,ly,lz,1))
		
		print "MAPGEN: Preparing pillaration table"
		self.p = np.ndarray(shape=(self.lx,self.ly,3), dtype=np.object_)
		
		print "MAPGEN: Opening VXL file %s" % repr(fname)
		fp = open(fname,"rb")
		
		print "MAPGEN: Parsing VXL"
		lzm = lz-1
		
		xst = 0
		xt = -offsy
		dont_skip_x = True
		for x in xrange(512):
			print x
			
			dont_skip_x = dont_skip_x and xt >= 0 and xt < lx
			
			yst = 0
			yt = -offsx
			dont_skip_y = True
			for y in xrange(512):
				dont_skip_y = dont_skip_y and dont_skip_x and yt >= 0 and yt < ly
				
				z = 0
				pl = []
				
				# read NSEA
				n,s,e,a = (ord(v) for v in fp.read(4))
				
				while True:
					# calculate correct N
					rn = e-s+1 if n == 0 else n-1
					
					#print n,s,e,a
					
					# load colour info
					cl = [fp.read(4) for i in xrange(rn)]
					
					# read new NSEA if necessary
					nn,ns,ne,na = 0,63,62,64
					if n != 0:
						nn,ns,ne,na = (ord(v) for v in fp.read(4))
					
					# skip if necessary
					if dont_skip_y:
						# add pillarisation data
						if s <= e:
							pl.append((lzm-e,lzm-s+1))
						
						# put air in
						if z < s:
							self.g[yt][xt][lzm-(s-1):lzm-z+1,3] = BTYPE_AIR
						z = s
						
						# put top colours in
						blk = None
						while z <= e:
							blk = b_poo_fail
							if cl:
								b,g,r,ign = (ord(v) for v in cl.pop(0))
								blk = to_rgb_t(r,g,b,BTYPE_SOLID)
							self.g[yt][xt][lzm-z] = blk
							z += 1
						
						# put bottom colours in
						k = 0 if n == 0 else rn - (e - s + 1)
						z = na - k
						while z < na:
							blk = b_poo_fail
							if cl:
								b,g,r,ign = (ord(v) for v in cl.pop(0))
								blk = to_rgb_t(r,g,b,BTYPE_SOLID)
							self.g[yt][xt][lzm-z] = blk
							z += 1
						
						# add pillarisation data
						if n != 0 and k != 0:
							pl.append((lzm-na+1,lzm-(na-k)+1))
						
						fail = True
						if cl:
							print "VXL BUG: Colours still remain! (%i)" % len(cl)
						#elif blk == None:
						#	print "VXL BUG: Unused span!"
						elif blk != None and tuple(blk) == tuple(b_poo_fail):
							print "VXL BUG: Missing colour info!"
						else:
							fail = False
						
						if fail:
							print "[%i, %i]: [%i %i %i %i]" % (x,y,n,s,e,a)
					
					# if N == 0, move on
					if n == 0:
						break
					
					# transfer new NSEA across
					n,s,e,a = nn,ns,ne,na
				
				# skip if necessary
				if dont_skip_y:
					# save pillarisation info
					self.p[yt][xt][0] = None
					self.p[yt][xt][1] = None
					self.p[yt][xt][2] = pl
				
				dont_skip_y = False
				yst += ly
				while yst > sizey:
					yt += 1
					yst -= sizey
					dont_skip_y = True
			
			dont_skip_x = False
			xst += lx
			while xst > sizex:
				xt += 1
				xst -= sizex
				dont_skip_x = True
		
		print "MAPGEN: Closing VXL"
		fp.close()
		
		print "MAPGEN: Fixing layer 0 (water)"
		flg = np.uint8(BTYPE_WATER | BTYPE_FLAG_UNBREAKABLE)
		self.g[:,:,0,3] = flg
		
		print "MAPGEN: Protecting layer 1 (solid)"
		flg = np.uint8(BTYPE_FLAG_UNBREAKABLE)
		#for x in xrange(lx):
		#	for y in xrange(ly):
		#		self.g[x][y][1][3] += flg
		self.g[:,:,1,3] |= flg
		
		if lx != sizex or ly != sizey:
			print "MAPGEN: HACK: removing pillaration (looks bad when scaled down!)"
			self.p = None
		
		print "MAPGEN: Done!"
	
	return _gen

class World:
	ready = False
	def __init__(self, lx, ly, lz, gen, win=None):
		self.lx = lx
		self.ly = ly
		self.lz = lz
		
		self.p = None
		gen(self, lx, ly, lz)
		self.pillarate()
		
		self.ready = True
	
	def test_if_air(self, x, y, z):
		return (
			z >= 0
			and x >= 0
			and x < self.lx
			and y >= 0
			and y < self.ly
			and (z >= self.lz
				or (int(self.g[x][y][z][3])&BTYPE_MASK_TYPE) == 0
			)
		)
	
	def test_if_solid(self, x, y, z):
		return (
			( 
				z < 0
				or x < 0
				or x >= self.lx
				or y < 0
				or y >= self.ly
				or (
					z < self.lz 
					and (int(self.g[x][y][z][3])&BTYPE_MASK_TYPE)
						in (BTYPE_SOLID,)
				)
			)
		)
	
	def test_if_visible(self, x, y, z):
		return (
			   self.test_if_air(x-1,y,z)
			or self.test_if_air(x+1,y,z)
			or self.test_if_air(x,y-1,z)
			or self.test_if_air(x,y+1,z)
			or self.test_if_air(x,y,z-1)
			or self.test_if_air(x,y,z+1)
		)
	
	def solid_check_box(self, x1, y1, z1, x2, y2, z2):
		for x in xrange(int(floor(min(x1,x2))),int(floor(max(x1,x2)-0.001+1))):
			for y in xrange(int(floor(min(y1,y2))),int(floor(max(y1,y2)-0.001+1))):
				for z in xrange(int(floor(min(z1,z2))),int(floor(max(z1,z2)-0.001+1))):
					if self.test_if_solid(x,y,z):
						return True
		
		return False
	
	def pillarate(self):
		if self.p != None:
			print "PILLARATE: World already pillarated - this should save some time :)"
			return
		
		print "PILLARATE: Generating z/vert air runs"
		self.p = np.ndarray(shape=(self.lx,self.ly,3), dtype=np.object_)
		for x in xrange(self.lx):
			print x
			for y in xrange(self.ly):
				l = []
				# (col_start, col_end) pairs
				# col_start <= z < col_end
				was_col = False
				col_start = -1
				col_end = -1
				for z in xrange(self.lz):
					v = self.g[x][y][z]
					r,g,b,t = v
					is_col = (int(t)&BTYPE_MASK_TYPE) != 0 and self.test_if_visible(x,y,z)
					if is_col:
						col_end = z
					else:
						if was_col:
							l.append((col_start+1, col_end+1))
						col_start = z
					was_col = is_col
				
				if was_col:
					l.append((col_start+1, col_end+1))
				
				#print l
				
				self.p[x][y][0] = None
				self.p[x][y][1] = None
				self.p[x][y][2] = l
		
		pass
	
	
	# NOTE: USE THIS FOR READ ONLY ACCESS ONLY
	def get_pillar_array(self):
		return self.p
	
	# NOTE: USE THIS FOR READ ONLY ACCESS ONLY
	def get_block_array(self):
		return self.g
	
