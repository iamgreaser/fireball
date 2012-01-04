#!/usr/bin/env python2 --

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

import sys

import numpy as np

import pyglet

import game, entity, render, world

USE_PSYCO = False

if USE_PSYCO:
	try:
		import psyco
		psyco.full()
		pyglet.options['debug_gl'] = False
	except ImportError:
		print "WARNING: Psyco not found. Continuing."
		print "Some stuff will be very slow though!"

#w = world.World(512,512,64,world.gen_aos_vxl("urbantankfight.vxl"))
#w = world.World(512,512,64,world.gen_aos_vxl("pinpoint.vxl"))
#w = world.World(512,512,64,world.gen_aos_vxl("stalingrad.vxl"))
#w = world.World(512,512,64,world.gen_aos_vxl("bridgewars.vxl"))
#w = world.World(512,512,64,world.gen_aos_vxl("mesa.vxl"))
#w = world.World(512,512,64,world.gen_aos_vxl("normandie.vxl"))
#w = world.World(128,128,64,world.gen_white(0.003))
#w = world.World(64,64,64,world.gen_white(0.01))
#w = world.World(256,256,64,world.gen_flat(3))

if len(sys.argv) > 1:
	w = world.World(512,512,64,world.gen_aos_vxl(sys.argv[1]))
else:
	w = world.World(64,64,64,world.gen_white(0.01))

g = game.LocalGame()
g.set_world(w)
p = entity.PlayerEntity(name="btrollface")
g.add_entity_main(p)
rend = render.Window()
rend.set_world(w)
rend.set_game(g)
rend.set_controller(p)
rend.mainloop()
