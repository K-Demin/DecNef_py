"""
 Copyright 2004-2024 BIOPAC Systems, Inc.

 This software is provided 'as-is', without any express or implied warranty.
 In no event will BIOPAC Systems, Inc. or BIOPAC Systems, Inc. employees be
 held liable for any damages arising from the use of this software.

 Permission is granted to anyone to use this software for any purpose,
 including commercial applications, and to alter it and redistribute it
 freely, subject to the following restrictions:

 1. The origin of this software must not be misrepresented; you must not
 claim that you wrote the original software. If you use this software in a
 product, an acknowledgment (see the following) in the product documentation
 is required.

 Portions Copyright 2004-2024 BIOPAC Systems, Inc.

 2. Altered source versions must be plainly marked as such, and must not be
 misrepresented as being the original software.

 3. This notice may not be removed or altered from any source distribution.

 spacegrip.py : This is a sample Python 3 program that makes use of the BIOPAC
      Hardware API to acquire data directly from MP device.  As written, the
      program will acquire data from an MP160 using the Research version of the
      Hardware API.  Other MP devices may be used with minor modification.

For all hardware types, note that pygame must be installed for this program to
run.  Use "pip3 install pygame" from a cmd window after installing Python in 
order to install pygame.  This program was last tested with Python 3.12.2 and 
pygame 2.5.2 on Windows 10 Professional.

"""
# necessary imports for use of BHAPI in Python 3
import mpenum
from ctypes import *

# imports used specifically in this example
import pygame
from pygame.locals import *
import sys
import time
import random

# load the mpdev DLL into memory (must be specified to correct location, including all its dependencies); change
# "Research" to "Education" if using MP36.  Program should function with SS25LB/SS25LA if using MP36
mpdev = cdll.LoadLibrary("C:/Program Files/BIOPAC Systems, Inc/BIOPAC Hardware API 2.2.5 Research/VC14/x64/mpdev.dll")

# attempt connection to MP device
print("\nAttempting to connect to MP device...\n")

# must pass c-specific types into function calls to BHAPI
mpdev.connectMPDev.argtypes = [c_int, c_int, c_char_p]

# modify first two arguments to use different hardware types; if using MP36, search for "diff" to find parameter to tweak
retval = mpdev.connectMPDev(mpenum.MP160, mpenum.MPUDP, b'auto') # use auto or match to serial number (ex. 1310A-00013BE)

# error occurs loading mpdev
if retval != mpenum.MPSUCCESS:
	print("FAILED: connectMPDev() with return code", retval,"\n")
	print("Disconnecting...\n")
	mpdev.disconnectMPDev()
	exit()

# successful connection to mpdev, begin application setup
print("Success! Beginning setup...\n")

# set sample rate to 500Hz
mpdev.setSampleRate.argtypes = [c_double]

retval = mpdev.setSampleRate(2.0)

if retval != mpenum.MPSUCCESS:
	print("FAILED: setSampleRate() with return code", retval,"\n")
	print("Disconnecting...\n")
	mpdev.disconnectMPDev()
	exit()

# set acquisition channel(s)
# for this example, we are setting channel 1 for connection to a Dynamometer (Grip Strength Meter)
#ex. channels[0] = True -> set acquisition from channel 1
#	 channels[1] = True -> set acquisition from channel 2 ... so on
arr_type = c_bool * 16
channels = arr_type(True, *[False]*15) # set channel 1

retval = mpdev.setAcqChannels(channels)

if retval != mpenum.MPSUCCESS:
	print("FAILED: setAcqChannels() with return code", retval,"\n")
	print("Disconnecting...\n")
	mpdev.disconnectMPDev()
	exit()

# start data acquisition from Dynamometer
retval = mpdev.startAcquisition()

if retval != mpenum.MPSUCCESS:
	print("FAILED: startAcquisition() with return code", retval,"\n")
	print("Disconnecting...\n")
	mpdev.stopAcquisition() # if failed, stop acquisition
	mpdev.disconnectMPDev()
	exit()

# returns current grip strength reading
def getGripStrength():

	# create array to receive sample data
	arr_type = c_double * 16
	samples = arr_type(0 * 16)

	# returns most recent value of analog channel(s)
	retval = mpdev.getMostRecentSample(samples)

	if retval == mpenum.MPSUCCESS:
		# samples[0] = channel 1, index should correspond to desired acquisition channel
		return samples[0]
	return None

# variable to store resting voltage of Dynamometer
initialVoltage = getGripStrength()

# detects and return a significant change in voltage
def compareVolts(volts):
	# find absolute difference of initial voltage to current reading
	diff = abs(initialVoltage - volts)
	# return diff only if significant change -- note that "significant" depends on hardware.  Testing with MP36R and SS25LA
        # required the value to be 0.00001.  Note that alternative could be to configure the device differently using other BHAPI commands
        # Search for "controlling" to find another parameter that might need to be adjusted
	if diff > 0.02:
		return diff
	return 0



############################################################################################



# [pygame section]
# "SpaceGrip" is a minigame that makes use of getGripStrength() to control a spaceship
print("Starting SpaceGrip...")

# application setup
pygame.init()
screen_width, screen_height = 1200, 600
win = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('SpaceGrip')

# global variables
global bg, bgX, bgX2, stars, starsX, starsX2

# set font types
startFont = pygame.font.SysFont('jokerman', 30)
font = pygame.font.SysFont('arial', 25, bold=True)

# set clock
clock = pygame.time.Clock()

# class for spaceship object
class spaceship(object):
	ship_img = pygame.image.load('dist/assets/ship.png').convert_alpha()
	ship_img = pygame.transform.scale(ship_img, (145, 60))
	def __init__(self, x, y, width, height):
		self.x = x
		self.y = y
		self.width = width
		self.height = height
		self.speed = 0
		self.acceleration = 0
		self.max_speed = 6
		self.crashed = False
	def draw(self, win):
		if self.crashed:
			pass
		else:
			self.hitbox_rect = pygame.Rect(self.x + 5, self.y + 20, self.width - 50, self.height - 40)
			win.blit(spaceship.ship_img, (self.x, self.y))
	# function for controlling spaceship movement when grip is held; note that there is a scale factor applied to "grip".  For
        # MP36R with SS25LA, this scale factor was changed to 500 (from the default (3.0)).  Some trial and error may be required.
	def control(self, grip):
		if grip > 0:
			self.acceleration = -(grip*3.0)
		else:
                        # establishes how fast the ship falls when no grip is applied
			self.acceleration = 0.2

		# accelerate
		self.speed += self.acceleration

		# if max speed is exceeded
		if abs(self.speed) >= self.max_speed:
			# normalize speed
			self.speed = self.speed / abs(self.speed) * self.max_speed

		# move spaceship
		self.y += self.speed

		# loop spaceship if it exits screen bounds
		if self.y < -60: self.y = screen_height + 60
		if self.y > screen_height + 60: self.y = -60

# create spaceship
ship = spaceship(screen_width/2-(200), screen_height/2-(60/2), 145, 60)

# class for space debris object
class debris(object):
	debris_img = pygame.image.load('dist/assets/debris.png').convert_alpha()
	# initialize debris
	def __init__(self, x, y, width, height):
		self.x = x
		self.y = y
		self.width = width
		self.height = height
		self.debris_img = pygame.transform.scale(debris.debris_img, (self.width, self.height))
	# draw debris
	def draw(self, win):
		self.hitbox_rect = pygame.Rect(self.x + 10, self.y + 5, self.width - 20, self.height - 5)
		win.blit(self.debris_img, (self.x, self.y))
	# for checking if debris has collided with the ship
	def collide(self, rect):
		if self.hitbox_rect.colliderect(rect): return True
		return False

# function for drawing text to the window
def draw_text(text, font, color, surface, x, y):
    textobj = font.render(text, 1, color)
    textrect = textobj.get_rect()
    textrect.topleft = ((screen_width/2-textrect.width/2)+x, (screen_height/2-textrect.height/2)+y)
    surface.blit(textobj, textrect)

# function for redrawing menu
def redrawMenu(logo1, logo2, logo1_box, logo2_box, spacebar):
	win.blit(logo1, logo1_box)
	win.blit(logo2, logo2_box)
	if spacebar == True:
		draw_text('PRESS SPACE TO PLAY', startFont, (255, 255, 255), win, 0, 90)

# function for redrawing the game window to support moving background
def redrawGame(bg, bgX, bgX2, stars, starsX, starsX2, hud, obstacles, hours, minutes, seconds, counter):
	win.blit(bg, (bgX, 0))  # draws our bg image
	win.blit(bg, (bgX2, 0))
	win.blit(stars, (starsX, 0)) #draw our star image
	win.blit(stars, (starsX2, 0))
	ship.draw(win)
	# obstacles
	for obstacle in obstacles:
		obstacle.draw(win)
		if obstacle.collide(ship.hitbox_rect):
			ship.crashed = True
	# hud
	hud_rect = hud.get_rect(center=(screen_width // 2, screen_height // 2 - 265))
	win.blit(hud, hud_rect)
	# control ship with reading from dynamometer
	grip = 0
	if counter == 0:
		grip = compareVolts(getGripStrength())
		ship.control(grip)
	# track grip value on-screen
	draw_text('GRIP: ' + str("%.2f"%grip), font, (255, 255, 255), win, 80, -267)
	# game timer
	draw_text('TIME: ' + str(minutes).rjust(2,'0') + ':' + str(seconds).rjust(2,'0'), font, (255, 255, 255), win, -75, -267)
	pygame.display.update()

def menu():
	counter = 0
	spacebar = True
	# biopac logo
	logo1 = pygame.image.load('dist/assets/logo1.png').convert_alpha()
	logo1_box = logo1.get_rect(center=(screen_width // 2, screen_height // 2 - 110))
	# spacegrip logo
	logo2 = pygame.image.load('dist/assets/logo2.png').convert_alpha()
	logo2_box = logo2.get_rect(center=(screen_width // 2, screen_height // 2 - 20))
	while True:
		win.fill((0, 0, 0))
		redrawMenu(logo1, logo2, logo1_box, logo2_box, spacebar)
		counter += 1
		if counter == 15:
			if spacebar == True: spacebar = False
			else: spacebar = True
			counter = 0

		for event in pygame.event.get():
			if event.type == QUIT:
				pygame.quit()
				sys.exit()
			if event.type == KEYDOWN:
				if event.key == K_ESCAPE:
					pygame.quit()
					sys.exit()
				if event.key == K_SPACE:
					game()

		pygame.display.update()
		clock.tick(60)

# game loop
def game():
	run = True
	rate = 1.5

	# load bg and stars
	bg = pygame.image.load('dist/assets/bg.png').convert_alpha()
	bgX = 0
	bgX2 = bg.get_width()
	stars = pygame.image.load('dist/assets/stars.png').convert_alpha()
	starsX = 0
	starsX2 = stars.get_width()

	# game hud
	hud = pygame.image.load('dist/assets/hud.png').convert_alpha()
	hud = pygame.transform.scale(hud, (520, 400))

	# stores obstacles
	obstacles = []

	# countdown font type
	countdownFont = pygame.font.SysFont('comicsansms', 50)

	# variables to track game timer
	hours = 0
	minutes = 0
	seconds = 0
	goCount = 0

	# 3 second countdown
	counter = 3
	while counter > 0:
		# background
		redrawGame(bg, bgX, bgX2, stars, starsX, starsX2, hud, obstacles, hours, minutes, seconds, counter)
		# countdown text
		draw_text('Grip in: ' + str(counter), countdownFont, (255, 255, 255), win, 100, 0)
		pygame.display.update()
		# sleep 1 second
		time.sleep(1)
		counter -= 1

	# sets the timer for 1 second
	pygame.time.set_timer(USEREVENT + 1, 1000)

	# timer for spawning debris obstacles
	pygame.time.set_timer(USEREVENT + 2, random.randrange(1000, 5000))
	pygame.time.set_timer(USEREVENT + 3, random.randrange(10000, 12000))

	# end of game counter
	end_counter = 5

	while run:
		# draw the pygame window
		redrawGame(bg, bgX, bgX2, stars, starsX, starsX2, hud, obstacles, hours, minutes, seconds, 0)

		if ship.crashed:
			rate = 0
			if end_counter == 0:
				ship.crashed = False
				ship.x = screen_width/2-(200)
				ship.y = screen_height/2-(60/2)
				pygame.time.set_timer(USEREVENT + 1, 0)
				pygame.time.set_timer(USEREVENT + 2, 0)
				pygame.time.set_timer(USEREVENT + 3, 0)
				run = False
			draw_text('You Crashed! Time: ' + str(minutes).rjust(2,'0') + ':' + str(seconds).rjust(2,'0'), countdownFont, (255, 255, 255), win, 0, -40)
			draw_text('Returning to Menu...' + str(end_counter), countdownFont, (255, 255, 255), win, 0, 20)
			pygame.display.update()

		if goCount <= 25:
			draw_text('GO!', countdownFont, (255, 255, 255), win, 100, 0)
			pygame.display.update()
			goCount += 1

		# move background images to the left
		bgX -= rate
		bgX2 -= rate
		starsX -= (rate + 1)
		starsX2 -= (rate + 1)

		# move obstacles
		for obstacle in obstacles:
			obstacle.x -= rate*1.5
			if obstacle.x < obstacle.width * -1:  # if our obstacle is off the screen we will remove it
				obstacles.pop(obstacles.index(obstacle))

		# reset background images when they run out
		if bgX < bg.get_width() * -1:
			bgX = bg.get_width()
		if bgX2 < bg.get_width() * -1:
			bgX2 = bg.get_width()
		if starsX < stars.get_width() * -1:
			starsX = stars.get_width()
		if starsX2 < stars.get_width() * -1:
			starsX2 = stars.get_width()

		# event loop
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
				pygame.quit()
				quit()
			if event.type == KEYDOWN:
				if event.key == K_ESCAPE:
					pygame.quit()
					sys.exit()
			# happens every 1 second
			if event.type == USEREVENT + 1 and ship.crashed == False:  # checks if timer goes off
				rate += .2
				seconds += 1
				if seconds == 60:
					minutes += 1
					seconds = 0
				if minutes == 60:
					hours += 1
					minutes = 0
			if event.type == USEREVENT + 1 and ship.crashed == True:
				end_counter -= 1
			# happens in random range to spawn space debris
			if event.type == USEREVENT + 2:
				x1 = random.randrange(0, screen_width) + 1200
				x2 = random.randrange(0, screen_width) + 1200
				y1 = random.randrange(0, screen_height // 2 - 100)
				y2 = random.randrange(screen_height // 2 + 100, screen_height)
				w1 = random.randrange(50, 125)
				h1 = random.randrange(50, 125)
				w2 = random.randrange(50, 125)
				h2 = random.randrange(50, 125)
				obstacles.append(debris(x1, y1, w1, h1))
				obstacles.append(debris(x2, y2, w2, h2))
			if event.type == USEREVENT + 3:
				x = random.randrange(0, screen_width) + 1200
				y = random.randrange(screen_height // 2 - 100, screen_height // 2 + 100)
				w = random.randrange(50, 125)
				h = random.randrange(50, 125)
				obstacles.append(debris(x, y, w, h))
		clock.tick(60)
	# return to menu when stops running
	menu()
# start menu
menu()
