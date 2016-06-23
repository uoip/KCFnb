import numpy as np
from numba.pycc import CC
import math


# constant
NUM_SECTOR = 9
FLT_EPSILON = 1e-07

cc = CC('fhog_utils')


@cc.export('func1', '(f4[:,:,:], f4[:,:,:], f4[:], f4[:], f4[:,:], i8[:,:,:], i8,i8,i8)')
def func1(dx, dy, boundary_x, boundary_y, r, alfa, height, width, numChannels):

    for j in xrange(1, height-1):
        for i in xrange(1, width-1):
            c = 0
            x = dx[j, i, c]
            y = dy[j, i, c]
            r[j, i] = math.sqrt(x*x + y*y)

            for ch in xrange(1, numChannels):
                tx = dx[j, i, ch]
                ty = dy[j, i, ch]
                magnitude = math.sqrt(tx*tx + ty*ty)
                if(magnitude > r[j, i]):
                    r[j, i] = magnitude
                    c = ch
                    x = tx
                    y = ty

            mmax = boundary_x[0]*x + boundary_y[0]*y
            maxi = 0

            for kk in xrange(0, NUM_SECTOR):
                dotProd = boundary_x[kk]*x + boundary_y[kk]*y
                if(dotProd > mmax):
                    mmax = dotProd
                    maxi = kk
                elif(-dotProd > mmax):
                    mmax = -dotProd
                    maxi = kk + NUM_SECTOR

            alfa[j, i, 0] = maxi % NUM_SECTOR
            alfa[j, i, 1] = maxi


@cc.export('func2', '(f4[:], f4[:], f4[:], f4[:,:], i8[:,:,:], i8[:], f4[:,:], i8,i8,i8,i8,i8,i8,i8)')
def func2(mapp, boundary_x, boundary_y, r, alfa, nearest, w, k, height, width, sizeX, sizeY, p, stringSize):
    
    for i in xrange(sizeY):
        for j in xrange(sizeX):
            for ii in xrange(k):
                for jj in xrange(k):
                    if((i * k + ii > 0) and (i * k + ii < height - 1) and (j * k + jj > 0) and (j * k + jj < width  - 1)):
                        mapp[i*stringSize + j*p + alfa[k*i+ii,j*k+jj,0]] +=  r[k*i+ii,j*k+jj] * w[ii,0] * w[jj,0]
                        mapp[i*stringSize + j*p + alfa[k*i+ii,j*k+jj,1] + NUM_SECTOR] +=  r[k*i+ii,j*k+jj] * w[ii,0] * w[jj,0]
                        if((i + nearest[ii] >= 0) and (i + nearest[ii] <= sizeY - 1)):
                            mapp[(i+nearest[ii])*stringSize + j*p + alfa[k*i+ii,j*k+jj,0]] += r[k*i+ii,j*k+jj] * w[ii,1] * w[jj,0]
                            mapp[(i+nearest[ii])*stringSize + j*p + alfa[k*i+ii,j*k+jj,1] + NUM_SECTOR] += r[k*i+ii,j*k+jj] * w[ii,1] * w[jj,0]
                        if((j + nearest[jj] >= 0) and (j + nearest[jj] <= sizeX - 1)):
                            mapp[i*stringSize + (j+nearest[jj])*p + alfa[k*i+ii,j*k+jj,0]] += r[k*i+ii,j*k+jj] * w[ii,0] * w[jj,1]
                            mapp[i*stringSize + (j+nearest[jj])*p + alfa[k*i+ii,j*k+jj,1] + NUM_SECTOR] += r[k*i+ii,j*k+jj] * w[ii,0] * w[jj,1]
                        if((i + nearest[ii] >= 0) and (i + nearest[ii] <= sizeY - 1) and (j + nearest[jj] >= 0) and (j + nearest[jj] <= sizeX - 1)):
                            mapp[(i+nearest[ii])*stringSize + (j+nearest[jj])*p + alfa[k*i+ii,j*k+jj,0]] += r[k*i+ii,j*k+jj] * w[ii,1] * w[jj,1]
                            mapp[(i+nearest[ii])*stringSize + (j+nearest[jj])*p + alfa[k*i+ii,j*k+jj,1] + NUM_SECTOR] += r[k*i+ii,j*k+jj] * w[ii,1] * w[jj,1]


@cc.export('func3', '(f4[:], f4[:], f4[:], i8,i8,i8,i8,i8)')
def func3(newData, partOfNorm, mappmap, sizeX, sizeY, p, xp, pp):
	
	for i in xrange(1, sizeY+1):
		for j in xrange(1, sizeX+1):
			pos1 = i * (sizeX+2) * xp + j * xp
			pos2 = (i-1) * sizeX * pp + (j-1) * pp

			valOfNorm = math.sqrt(partOfNorm[(i    )*(sizeX + 2) + (j    )] +
                				partOfNorm[(i    )*(sizeX + 2) + (j + 1)] +
                				partOfNorm[(i + 1)*(sizeX + 2) + (j    )] +
                				partOfNorm[(i + 1)*(sizeX + 2) + (j + 1)]) + FLT_EPSILON
			newData[pos2:pos2+p] = mappmap[pos1:pos1+p] / valOfNorm
			newData[pos2+4*p:pos2+6*p] = mappmap[pos1+p:pos1+3*p] / valOfNorm

			valOfNorm = math.sqrt(partOfNorm[(i    )*(sizeX + 2) + (j    )] +
				                partOfNorm[(i    )*(sizeX + 2) + (j + 1)] +
				                partOfNorm[(i - 1)*(sizeX + 2) + (j    )] +
				                partOfNorm[(i - 1)*(sizeX + 2) + (j + 1)]) + FLT_EPSILON
			newData[pos2+p:pos2+2*p] = mappmap[pos1:pos1+p] / valOfNorm
			newData[pos2+6*p:pos2+8*p] = mappmap[pos1+p:pos1+3*p] / valOfNorm

			valOfNorm = math.sqrt(partOfNorm[(i    )*(sizeX + 2) + (j    )] +
				                partOfNorm[(i    )*(sizeX + 2) + (j - 1)] +
				                partOfNorm[(i + 1)*(sizeX + 2) + (j    )] +
				                partOfNorm[(i + 1)*(sizeX + 2) + (j - 1)]) + FLT_EPSILON
			newData[pos2+2*p:pos2+3*p] = mappmap[pos1:pos1+p] / valOfNorm
			newData[pos2+8*p:pos2+10*p] = mappmap[pos1+p:pos1+3*p] / valOfNorm

			valOfNorm = math.sqrt(partOfNorm[(i    )*(sizeX + 2) + (j    )] +
				                partOfNorm[(i    )*(sizeX + 2) + (j - 1)] +
				                partOfNorm[(i - 1)*(sizeX + 2) + (j    )] +
				                partOfNorm[(i - 1)*(sizeX + 2) + (j - 1)]) + FLT_EPSILON
			newData[pos2+3*p:pos2+4*p] = mappmap[pos1:pos1+p] / valOfNorm
			newData[pos2+10*p:pos2+12*p] = mappmap[pos1+p:pos1+3*p] / valOfNorm


@cc.export('func4', '(f4[:], f4[:], i8,i8,i8,i8,i8,i8,f8,f8)')
def func4(newData, mappmap, p, sizeX, sizeY, pp, yp, xp, nx, ny):
	
	for i in xrange(sizeY):
		for j in xrange(sizeX):
			pos1 = (i*sizeX + j) * p
			pos2 = (i*sizeX + j) * pp

			for jj in xrange(2 * xp):  # 2*9
				newData[pos2 + jj] = np.sum(mappmap[pos1 + yp*xp + jj : pos1 + 3*yp*xp + jj : 2*xp]) * ny
			for jj in xrange(xp):  # 9
				newData[pos2 + 2*xp + jj] = np.sum(mappmap[pos1 + jj : pos1 + jj + yp*xp : xp]) * ny
			for ii in xrange(yp):  # 4
				newData[pos2 + 3*xp + ii] = np.sum(mappmap[pos1 + yp*xp + ii*xp*2 : pos1 + yp*xp + ii*xp*2 + 2*xp]) * nx
				
	
	
if __name__ == "__main__":
    cc.compile()
