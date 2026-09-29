def matrix_tiles(matrix, rows, columns):
    height, width = matrix.shape
    return (
        matrix.reshape(height // rows, rows, width // columns, columns)
        .transpose(0, 2, 1, 3)
        .copy()
    )
