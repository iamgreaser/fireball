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

import numpy

import helpers

class AbstractEvent(helpers.ArgGenerator):
	ARGS = ["timestamp"]
	IDENT = "ev"

class ControllerEvent(AbstractEvent):
	ARGS = AbstractEvent.ARGS + ["idx","keys"]
	IDENT = AbstractEvent.IDENT + "-controller"
	
	def apply_to_entity(self, ent):
		ent.set_keys(self.keys)

class OrientationEvent(AbstractEvent):
	ARGS = AbstractEvent.ARGS + ["idx","origin","velocity","orient_x","orient_y"]
	IDENT = AbstractEvent.IDENT + "-orientation"
	
	def apply_to_entity(self, ent):
		ent.set_orient(self.origin, self.velocity, self.orient_x, self.orient_y)

# TODO: set this up properly
class ShootEvent(AbstractEvent):
	ARGS = AbstractEvent.ARGS + ["direction"]
	IDENT = AbstractEvent.IDENT + "-shoot"
