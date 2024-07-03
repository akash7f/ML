def sortbycol(A, Col):
    """
    A is a multidimensional array of more than one dimension
    Col is the reference to the column by which the A have to be sort out
    """
    n = len(A)
    if n < 64:                      #insertion sort
        for i in range(1, n):
            key_row = A[i, :].copy()
            key = key_row[Col]
            j = i - 1
            while j >= 0 and key < A[j, Col]:
                A[j + 1, :] = A[j, :]
                j -= 1
            A[j + 1, :] = key_row
    else:                           #merge sort
        mid = n//2
        left, l_n = A[:mid, :].copy(), mid
        right, r_n = A[mid:, :].copy(), n - mid
        sortbycol(left, Col)
        sortbycol(right, Col)

        i, j = 0, 0
        while i < l_n and j < r_n:
            if left[i, Col] < right[j, Col]:
                A[i+j, :] = left[i, :]
                i += 1
            else:
                A[i+j, :] = right[j, :]
                j += 1

        while i < l_n:
            A[i+j, :] = left[i, :]
            i += 1

        while j < r_n:
            A[i+j, :] = right[j, :]
            j += 1

def sortbyrow(A, Row):
    """
    A is a multidimensional array of more than one dimension
    Row is the reference to the row by which the A have to be sort out
    """
    n = len(A[0])
    if n < 64:                      #insertion sort
        for i in range(1, n):
            key_col = A[:, i].copy()
            key = key_col[Row]
            j = i - 1
            while j >= 0 and key < A[Row, j]:
                A[:, j + 1] = A[:, j]
                j -= 1
            A[:, j + 1] = key_col
    else:                           #merge sort
        mid = n//2
        left, l_n = A[:, :mid].copy(), mid
        right, r_n = A[:, mid:].copy(), n - mid
        sortbyrow(left, Row)
        sortbyrow(right, Row)

        i, j = 0, 0
        while i < l_n and j < r_n:
            if left[Row, i] < right[Row, j]:
                A[:, i+j] = left[:, i]
                i += 1
            else:
                A[:, i+j] = right[:, j]
                j += 1

        while i < l_n:
            A[:, i+j] = left[:, i]
            i += 1

        while j < r_n:
            A[:, i+j] = right[:, j]
            j += 1


if __name__ == "__main__":
    import numpy
    x = numpy.array([[1, 2, 3], [4, 8, 6], [7, 5, 9],[1, 1, 3], [4, 3, 6], [7, 4, 9]])
    sortbycol(x, 1)
    print(x)
    sortbyrow(x, 2)
    print(x)