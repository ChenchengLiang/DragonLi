import cupy as cp



def check_cuda():

    try:

        # Attempt to create a CuPy array on the GPU

        cp.array([1, 2, 3])

        print("CUDA is available!")

    except cp.cuda.runtime.CUDARuntimeError as e:

        print("CUDA is not available: ", e)



if __name__ == "__main__":

    check_cuda()

