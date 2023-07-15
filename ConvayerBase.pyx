import time
import numpy
cimport numpy
cimport cython
from libc.math cimport cos

cdef float pi = 3.14159


ctypedef numpy.int32_t DTYPE_int32
ctypedef numpy.uint8_t DTYPE_uint8
ctypedef numpy.float32_t DTYPE_float32

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def moving_avrage(numpy.ndarray[DTYPE_int32, ndim=1] arr, int window):   # This function is used to apply the moving-average function 

    cdef long int total
    cdef int i,w
    cdef int arr_shape = arr.shape[0]
    cdef numpy.ndarray[DTYPE_int32, ndim=1] res = numpy.zeros((arr_shape - window,), dtype = numpy.int32)
    #res = numpy.zeros((arr_shape - window,), dtype = numpy.int32 )

    for i in range(arr_shape - window):
        total = 0
        for w in range(window):
            total += arr[i+w]
        total = total / window
        res[i] = total
    
    return res



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def moving_avrage_float(numpy.ndarray[DTYPE_float32, ndim=1] arr, int window):

    cdef float total
    cdef int i,w
    cdef int arr_shape = arr.shape[0]
    cdef numpy.ndarray[DTYPE_float32, ndim=1] res = numpy.zeros((arr_shape - window,), dtype = numpy.float32)
    #res = numpy.zeros((arr_shape - window,), dtype = numpy.int32 )

    for i in range(arr_shape - window):
        total = 0
        for w in range(window):
            total += arr[i+w]
        total = total / window
        res[i] = total
    
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
def extract_points(numpy.ndarray[DTYPE_uint8, ndim=2] img, int thresh, int perspective_angle):  ##The one-dimensional transformation occurs assigning to each column j of the image I the mean
value of all the indices i of each row when I(i, j) > thresh

    cdef long long int total_sum
    cdef int total_count, last_j
    cdef int i,j, point_idx = 0
    cdef int img_h = img.shape[0]
    cdef int img_w = img.shape[1]
    cdef numpy.ndarray[DTYPE_int32, ndim=2] res_pts = numpy.zeros( (img_w, 2), dtype = numpy.int32 ) 
    cdef float perspective = cos(perspective_angle * pi / 180)     

    for i in range(img_w):
        total_sum = 0
        total_count = 0
        for j in range(img_h):
            if img[j,i] > thresh:
                
                # remove noise 
                if total_count>0 and total_count<3 and j - last_j > 5:
                    total_count = 1
                    total_sum = j
                    last_j = j
                else:
                    total_count +=1
                    total_sum += j
                    last_j = j
        
        
        if total_count>0:
            res_pts[point_idx,0] = i
            res_pts[point_idx,1] = int((total_sum / total_count) / perspective)
            point_idx+=1


    
    return res_pts[:point_idx]


@cython.boundscheck(False)
@cython.wraparound(False)
def extract_points_left(numpy.ndarray[DTYPE_uint8, ndim=2] img, int thresh, int perspective_angle ):

    cdef int total_count, last_j
    cdef int i,j, point_idx = 0
    cdef int img_h = img.shape[0]
    cdef int img_w = img.shape[1]
    cdef numpy.ndarray[DTYPE_int32, ndim=2] res_pts = numpy.zeros( (img_w, 2), dtype = numpy.int32 ) 
    cdef float perspective = cos(perspective_angle * pi / 180)



    for i in range(img_h):
        total_count = 0
        for j in range(img_w):
            if img[i,j] > thresh:
                
                if total_count == 0:
                    last_j = j
                    total_count+=1
                
                elif total_count > 0 and total_count < 3 and j - last_j > 5:
                    total_count = 1
                    last_j = j
                else:
                    total_count +=1
                    last_j = j
                
                if total_count>=8:
                    break


        
        if total_count>0:
            res_pts[point_idx,0] = int(last_j / perspective)
            res_pts[point_idx,1] = i
            point_idx+=1


    
    return res_pts[:point_idx]



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def calc_slope(numpy.ndarray[DTYPE_int32, ndim=2] arr, int step):   # This function is used to calculate slope for the input array

    cdef float slope
    cdef int i = 0
    cdef int arr_shape = arr.shape[0]
    cdef numpy.ndarray[DTYPE_float32, ndim=1] res = numpy.zeros((arr_shape - step,), dtype = numpy.float32)

    for i in range(arr_shape - step):
        slope = ( arr[ i + step, 1] - arr[i, 1] ) / (( arr[ i + step, 0] - arr[i ,0] ) + 0.001 ) 
        #res[i,0] = arr[i,0]
        #slope = ( 3*arr[ i-3, 1]+3*arr[ i-2, 1]-4*arr[i-1, 1] -4*arr[ i , 1] -4*arr[ i+1, 1]+3*arr[ i+2, 1]+3*arr[i+3, 1] ) / (( 3*arr[ i-3, 0]+3*arr[ i-2, 0]-4*arr[i-1, 0] -4*arr[ i , 0] -4*arr[ i+1,0]+3*arr[ i+2, 0]+3*arr[i+3, 0] ) + 0.001 ) 
        res[i] = slope
        print(slope)
        

    return res

