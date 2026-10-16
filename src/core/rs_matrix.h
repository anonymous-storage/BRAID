#ifndef BRAID_CORE_RS_MATRIX_H_
#define BRAID_CORE_RS_MATRIX_H_

#include <cstdint>
#include <vector>

namespace braid::core {

using Matrix = std::vector<std::vector<uint8_t>>;

Matrix BuildSystematicGeneratorMatrix(int k, int r);
Matrix InvertMatrix(const Matrix& matrix);
Matrix MultiplyMatrices(const Matrix& lhs, const Matrix& rhs);
Matrix SelectRows(const Matrix& matrix, const std::vector<int>& row_indices);
Matrix BuildRecoveryMatrix(
    const Matrix& generator,
    const std::vector<int>& survivor_indices,
    const std::vector<int>& target_indices);

}  // namespace braid::core

#endif  // BRAID_CORE_RS_MATRIX_H_

