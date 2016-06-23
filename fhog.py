import numpy as np 
import cv2
from fhog_utils import func1,func2,func3,func4

# constant
NUM_SECTOR = 9
FLT_EPSILON = 1e-07



def getFeatureMaps(image, k, mapp):
	kernel = np.array([[-1.,  0., 1.]], np.float32)

	height = image.shape[0]
	width = image.shape[1]
	assert(image.ndim==3 and image.shape[2])
	numChannels = 3 #(1 if image.ndim==2 else image.shape[2])

	sizeX = width / k
	sizeY = height / k
	px = 3 * NUM_SECTOR
	p = px
	stringSize = sizeX * p

	mapp['sizeX'] = sizeX
	mapp['sizeY'] = sizeY
	mapp['numFeatures'] = p
	mapp['map'] = np.zeros((mapp['sizeX']*mapp['sizeY']*mapp['numFeatures']), np.float32)

	dx = cv2.filter2D(np.float32(image), -1, kernel)   # np.float32(...) is necessary
	dy = cv2.filter2D(np.float32(image), -1, kernel.T)

	arg_vector = np.arange(NUM_SECTOR+1).astype(np.float32) * np.pi / NUM_SECTOR
	boundary_x = np.cos(arg_vector) 
	boundary_y = np.sin(arg_vector)

	###
	r = np.zeros((height, width), np.float32)
	alfa = np.zeros((height, width, 2), np.int)
	func1(dx, dy, boundary_x, boundary_y, r, alfa, height, width, numChannels) #with @jit
	### ~0.001s

	nearest = np.ones((k), np.int)
	nearest[0:k/2] = -1

	w = np.zeros((k, 2), np.float32)
	a_x = np.concatenate((k/2 - np.arange(k/2) - 0.5, np.arange(k/2,k) - k/2 + 0.5)).astype(np.float32)
	b_x = np.concatenate((k/2 + np.arange(k/2) + 0.5, -np.arange(k/2,k) + k/2 - 0.5 + k)).astype(np.float32)
	w[:, 0] = 1.0 / a_x * ((a_x*b_x) / (a_x+b_x))
	w[:, 1] = 1.0 / b_x * ((a_x*b_x) / (a_x+b_x))

	###
	mappmap = np.zeros((sizeX*sizeY*p), np.float32)
	func2(mappmap, boundary_x, boundary_y, r, alfa, nearest, w, k, height, width, sizeX, sizeY, p, stringSize)
	mapp['map'] = mappmap
	### ~0.001s
	

	return mapp


def normalizeAndTruncate(mapp, alfa):
	sizeX = mapp['sizeX']
	sizeY = mapp['sizeY']

	p = NUM_SECTOR
	xp = NUM_SECTOR * 3
	pp = NUM_SECTOR * 12

	'''
	### original implementation
	partOfNorm = np.zeros((sizeY*sizeX), np.float32)

	for i in xrange(sizeX*sizeY):
		pos = i * mapp['numFeatures']
		partOfNorm[i] = np.sum(mapp['map'][pos:pos+p]**2) ###
	'''
	### 50x speedup
	idx = np.arange(0, sizeX*sizeY*mapp['numFeatures'], mapp['numFeatures']).reshape((sizeX*sizeY, 1)) + np.arange(p)
	partOfNorm = np.sum(mapp['map'][idx] ** 2, axis=1) ### ~0.0002s

	sizeX, sizeY = sizeX-2, sizeY-2
	
	### 
	newData = np.zeros((sizeY*sizeX*pp), np.float32)
	func3(newData, partOfNorm, mapp['map'], sizeX, sizeY, p, xp, pp) #with @jit
	###

	# truncation
	newData[newData > alfa] = alfa

	mapp['numFeatures'] = pp
	mapp['sizeX'] = sizeX
	mapp['sizeY'] = sizeY
	mapp['map'] = newData

	return mapp


def PCAFeatureMaps(mapp):
	sizeX = mapp['sizeX']
	sizeY = mapp['sizeY']

	p = mapp['numFeatures']
	pp = NUM_SECTOR * 3 + 4
	yp = 4
	xp = NUM_SECTOR

	nx = 1.0 / np.sqrt(xp*2)
	ny = 1.0 / np.sqrt(yp)
	### 
	newData = np.zeros((sizeX*sizeY*pp), np.float32)
	func4(newData, mapp['map'], p, sizeX, sizeY, pp, yp, xp, nx, ny) #with @jit
	###

	mapp['numFeatures'] = pp
	mapp['map'] = newData

	return mapp
