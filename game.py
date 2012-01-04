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

import world, event, entity

class AbstractGame:
	def __init__(self, *args, **kwargs):
		self.entities_main = [None for i in xrange(256)]
		self.entities_main_free = set([i for i in xrange(len(self.entities_main))])
		self.entities_anon = set()
	
	def set_world(self, world):
		self.world = world
	
	def add_entity_main(self, ent, idx=None):
		# TODO: catch some errors
		if idx == None:
			idx = self.entities_main_free.pop()
		else:
			self.entities_main_free.remove(idx)
		
		self.entities_main[idx] = ent
		
		ent.set_game(idx, self)
	
	def rm_entity_main(self, idx):
		ent = self.entities_main[idx]
		if ent == None:
			return
		
		ent.set_game(-1, None)
		
		self.entities_main[idx] = None
		self.entities_main_free.add(idx)
	
	def update(self, dt):
		for ent in self.entities_main:
			if ent == None:
				continue
			
			ent.update(dt)
		
		for ent in self.entities_anon:
			ent.update(dt)

class LocalGame(AbstractGame):
	def __init__(self, *args, **kwargs):
		AbstractGame.__init__(self, *args, **kwargs)
		pass
	
	def send_event(self, idx, event):
		self.recv_event(idx, event)
	
	def recv_event(self, idx, event):
		if idx == None:
			ent = self.entities_main[idx]
			if ent != None:
				ent.handle_event(event)
		else:
			for ent in self.entities_main:
				if ent == None:
					continue
				
				ent.handle_event()
			
			for ent in self.entities_anon:
				ent.handle_event()

class NetworkGame(AbstractGame):
	def __init__(self, *args, **kwargs):
		AbstractGame.__init__(self, *args, **kwargs)
		pass

class ServerGame(NetworkGame):
	def __init__(self, *args, **kwargs):
		NetworkGame.__init__(self, *args, **kwargs)
		pass

class ClientGame(NetworkGame):
	def __init__(self, *args, **kwargs):
		NetworkGame.__init__(self, *args, **kwargs)
		pass
