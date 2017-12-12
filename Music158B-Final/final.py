import argparse
import math
import random
import time
import numpy as np
from collections import deque
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import osc_message_builder
from pythonosc import udp_client

pt = time.monotonic()
evolve = False
generate_state = False
class Model():
    def __init__(self, max_duration=2.0, note_depth=3, time_depth=3, time_divisions=15, note_rate=0.1, time_rate=0.1, initialization_factor=1000):
        self.initialization_factor = initialization_factor
        self.note_prob = np.random.random([5] * note_depth) / self.initialization_factor
        self.time_prob = np.random.random([time_divisions] * time_depth) / self.initialization_factor
        self.prev_notes = deque(np.random.choice(5, note_depth), note_depth)
        self.prev_times = deque(np.random.choice(time_divisions, time_depth), time_depth)
        self.note_rate = note_rate
        self.time_rate = time_rate
        self.max_duration = max_duration
        self.time_divisions = time_divisions
        self.time_duration = max_duration / time_divisions

    def add_sample(self, note, duration):
        self.prev_notes.append(note)
        self.note_prob[tuple(self.prev_notes)] += self.note_rate
        self.prev_notes.pop()
        self.note_prob[tuple(self.prev_notes)] /= np.sum(self.note_prob[tuple(self.prev_notes)])
        self.prev_notes.append(note)

        self.prev_times.append(duration)
        self.time_prob[tuple(self.prev_times)] += self.time_rate
        self.prev_times.pop()
        self.time_prob[tuple(self.prev_times)] /= np.sum(self.time_prob[tuple(self.prev_times)])
        self.prev_times.append(duration)

    def generate_note(self):
        self.prev_notes.popleft()
        note = np.random.choice(5, p=(self.note_prob[tuple(self.prev_notes)] / np.sum(self.note_prob[tuple(self.prev_notes)])))
        self.prev_notes.append(note)
        return note

    def generate_duration(self):
        self.prev_times.popleft()
        duration = np.random.choice(self.time_divisions, p=(self.time_prob[tuple(self.prev_times)] / np.sum(self.time_prob[tuple(self.prev_times)])))
        self.prev_times.append(duration)
        return abs(np.random.normal(duration * self.time_duration, self.time_duration / 2.0))

    def clear(self):
        self.note_prob = np.random.random(self.note_prob.shape) / self.initialization_factor
        self.time_prob = np.random.random(self.time_prob.shape) / self.initialization_factor
        self.prev_notes = deque(np.random.choice(5, self.prev_notes.maxlen), self.prev_notes.maxlen)
        self.prev_times = deque(np.random.choice(self.time_divisions, self.prev_times.maxlen), self.prev_times.maxlen)

    def reset_note_depth(self, depth):
        self.note_prob = np.random.random([5] * depth) / self.initialization_factor
        self.prev_notes = deque(np.random.choice(5, depth), depth)

    def reset_time_depth(self, depth):
        self.time_prob = np.random.random([self.time_divisions] * depth) / self.initialization_factor
        self.prev_times = deque(np.random.choice(self.time_divisions, depth), depth)

    def set_note_rate(self, rate):
        self.note_rate = rate

    def set_time_rate(self, rate):
        self.time_rate = rate

    def set_max_duration(self, duration):
        self.max_duration = duration
        self.time_duration = duration / self.time_divisions

    def set_time_divisions(self, divisions):
        self.time_divisions = divisions
        self.time_prob = np.random.random([divisions] * self.prev_times.maxlen) / self.initialization_factor
        self.prev_times = deque(np.random.choice(divisions, self.prev_times.maxlen), self.prev_times.maxlen)
        self.time_duration = self.max_duration / divisions

def note_handler(addr, args, note):
    global generate_state, pt
    generate_state = False
    n = int(note)
    client = args[0]
    model = args[1]
    t = time.monotonic()
    d = t - pt
    if d > model.max_duration:
        d = int(model.prev_times[0])
    else:
        d = int(d / model.max_duration)
    client.send_message("/note", n)
    model.add_sample(n, d)
    pt = t

def generate_handler(addr, args, state):
    global generate_state, evolve
    if (state == 1):
        generate_state = True
    else:
        generate_state = False
    client = args[0]
    model = args[1]
    while generate_state:
        note = model.generate_note()
        client.send_message("/note", note)
        duration = model.generate_duration()
        time.sleep(duration)
        if evolve:
            model.add_sample(note, int(duration / model.max_duration))

def evolve_mode(addr, state):
    global evolve
    if (state == 1):
        evolve = True
    else:
        evolve = False

def clear_handler(addr, args, state):
    args[0].clear()

def reset_note_depth_handler(addr, args, depth):
    args[0].reset_note_depth(int(depth) + 1)

def reset_time_depth_handler(addr, args, depth):
    args[0].reset_time_depth(int(depth) + 1)

def note_rate_handler(addr, args, rate):
    args[0].set_note_rate(rate)

def time_rate_handler(addr, args, rate):
    args[0].set_time_rate(rate)

def max_duration_handler(addr, args, duration):
    args[0].set_max_duration(duration * 4.9 + 0.1)

def time_divisions_handler(addr, args, divisions):
    args[0].set_time_divisions(int(divisions) + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip",
      default="169.254.187.231", help="The ip to listen on")
    parser.add_argument("--s_port",
      type=int, default=8000, help="The server port to listen on")
    parser.add_argument("--c_port", type=int, default=8001,
      help="The port the client sends to")
    args = parser.parse_args()

    client = udp_client.SimpleUDPClient(args.ip, args.c_port)
    model = Model()

    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/note", note_handler, client, model)
    dispatcher.map("/generate", generate_handler, client, model)
    dispatcher.map("/evolve", evolve_mode)
    dispatcher.map("/clear", clear_handler, model)
    dispatcher.map("/note_depth", reset_note_depth_handler, model)
    dispatcher.map("/time_depth", reset_time_depth_handler, model)
    dispatcher.map("/note_rate", note_rate_handler, model)
    dispatcher.map("/time_rate", time_rate_handler, model)
    dispatcher.map("/max_duration", max_duration_handler, model)
    dispatcher.map("/time_divisions", time_divisions_handler, model)

    server = osc_server.ThreadingOSCUDPServer(
      (args.ip, args.s_port), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()
