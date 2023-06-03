import numpy as np

"""
The Hungarian Algorithm helps us arrive at the final optimal assignment od labels to the detected objects 
using a few steps that involve row and column reduction on the adjacency matrix.
"""
import numpy as np

class Hungarian(object):
    def __init__(self, arr_costs):
        self.X = arr_costs.copy()
        n, m = self.X.shape
        self.u_row = np.ones(n, dtype=bool)
        self.u_column = np.ones(m, dtype=bool)
        self.r_0Z = 0
        self.c_0Z = 0
        self.course = np.zeros((n + m, 2), dtype=int)
        self.check = np.zeros((n, m), dtype=int)

    def clear(self):
        self.u_row[:] = True
        self.u_column[:] = True

    def row_reduction(self):
        self.X -= self.X.min(axis=1)[:, np.newaxis]
        for i, j in zip(*np.where(self.X == 0)):
            if self.u_column[j] and self.u_row[i]:
                self.check[i, j] = 1
                self.u_column[j] = False
                self.u_row[i] = False
        self.clear()
        return self.cover_columns

    def cover_columns(self):
        check = self.check == 1
        self.u_column[np.any(check, axis=0)] = False
        if check.sum() < self.X.shape[0]:
            return self.cover_zeros

    def cover_zeros(self):
        X = (self.X == 0).astype(int)
        covered = X * self.u_row[:, np.newaxis]
        covered *= np.asarray(self.u_column, dtype=int)
        n = self.X.shape[0]
        m = self.X.shape[1]

        while True:
            row, col = np.unravel_index(np.argmax(covered), (n, m))
            if covered[row, col] == 0:
                return self.generate_zeros
            else:
                self.check[row, col] = 2
                star_col = np.argmax(self.check[row] == 1)
                if self.check[row, star_col] != 1:
                    self.r_0Z = row
                    self.c_0Z = col
                    count = 0
                    course = self.course
                    course[count, 0] = self.r_0Z
                    course[count, 1] = self.c_0Z

                    while True:
                        row = np.argmax(self.check[:, course[count, 1]] == 1)
                        if self.check[row, course[count, 1]] != 1:
                            break
                        else:
                            count += 1
                            course[count, 0] = row
                            course[count, 1] = course[count - 1, 1]

                        col = np.argmax(self.check[course[count, 0]] == 2)
                        if self.check[row, col] != 2:
                            col = -1
                        count += 1
                        course[count, 0] = course[count - 1, 0]
                        course[count, 1] = col

                    for i in range(count + 1):
                        if self.check[course[i, 0], course[i, 1]] == 1:
                            self.check[course[i, 0], course[i, 1]] = 0
                        else:
                            self.check[course[i, 0], course[i, 1]] = 1

                    self.clear()
                    self.check[self.check == 2] = 0
                    return self.cover_columns
                else:
                    col = star_col
                    self.u_row[row] = False
                    self



def get_minimum_cost_assignment(arr_costs):
    arr_costs = np.asarray(arr_costs)

    if arr_costs.shape[1] < arr_costs.shape[0]:
        arr_costs = arr_costs.T
        is_T = True
    else:
        is_T = False

    assignment = Hungarian(arr_costs)

    run = assignment.row_reduction if 0 in arr_costs.shape else assignment.cover_columns

    while run is not None:
        run = run()

    if is_T:
        check = assignment.check.T
    else:
        check = assignment.check
    return np.where(check == 1)