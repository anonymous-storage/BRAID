#include "src/core/rs_matrix.h"

#include <stdexcept>

#include "src/core/gf256.h"

namespace braid::core {

namespace {

Matrix IdentityMatrix(size_t n) {
  Matrix identity(n, std::vector<uint8_t>(n, 0));
  for (size_t i = 0; i < n; ++i) {
    identity[i][i] = 1;
  }
  return identity;
}

}  // namespace

Matrix MultiplyMatrices(const Matrix& lhs, const Matrix& rhs) {
  if (lhs.empty() || rhs.empty() || lhs.front().size() != rhs.size()) {
    throw std::invalid_argument("matrix dimension mismatch");
  }
  Matrix out(lhs.size(), std::vector<uint8_t>(rhs.front().size(), 0));
  for (size_t i = 0; i < lhs.size(); ++i) {
    for (size_t k = 0; k < rhs.size(); ++k) {
      if (lhs[i][k] == 0) {
        continue;
      }
      for (size_t j = 0; j < rhs.front().size(); ++j) {
        out[i][j] ^= GF256::Multiply(lhs[i][k], rhs[k][j]);
      }
    }
  }
  return out;
}

Matrix InvertMatrix(const Matrix& matrix) {
  if (matrix.empty() || matrix.size() != matrix.front().size()) {
    throw std::invalid_argument("matrix must be square");
  }
  const size_t n = matrix.size();
  Matrix a = matrix;
  Matrix inv = IdentityMatrix(n);
  for (size_t col = 0; col < n; ++col) {
    size_t pivot = col;
    while (pivot < n && a[pivot][col] == 0) {
      ++pivot;
    }
    if (pivot == n) {
      throw std::runtime_error("matrix is singular");
    }
    if (pivot != col) {
      std::swap(a[pivot], a[col]);
      std::swap(inv[pivot], inv[col]);
    }
    const uint8_t pivot_value = a[col][col];
    const uint8_t inv_pivot = GF256::Inverse(pivot_value);
    for (size_t j = 0; j < n; ++j) {
      a[col][j] = GF256::Multiply(a[col][j], inv_pivot);
      inv[col][j] = GF256::Multiply(inv[col][j], inv_pivot);
    }
    for (size_t row = 0; row < n; ++row) {
      if (row == col || a[row][col] == 0) {
        continue;
      }
      const uint8_t factor = a[row][col];
      for (size_t j = 0; j < n; ++j) {
        a[row][j] ^= GF256::Multiply(factor, a[col][j]);
        inv[row][j] ^= GF256::Multiply(factor, inv[col][j]);
      }
    }
  }
  return inv;
}

Matrix SelectRows(const Matrix& matrix, const std::vector<int>& row_indices) {
  Matrix out;
  out.reserve(row_indices.size());
  for (int row : row_indices) {
    if (row < 0 || row >= static_cast<int>(matrix.size())) {
      throw std::invalid_argument("row index out of range");
    }
    out.push_back(matrix[row]);
  }
  return out;
}

Matrix BuildSystematicGeneratorMatrix(int k, int r) {
  if (k <= 0 || r <= 0 || (k + r) > 255) {
    throw std::invalid_argument("invalid rs dimensions");
  }
  const int n = k + r;
  std::vector<uint8_t> points(n);
  for (int i = 0; i < n; ++i) {
    points[i] = GF256::Pow(GF256::kPrimitive, i);
  }

  Matrix vandermonde(n, std::vector<uint8_t>(k, 0));
  for (int row = 0; row < n; ++row) {
    uint8_t value = 1;
    for (int col = 0; col < k; ++col) {
      vandermonde[row][col] = value;
      value = GF256::Multiply(value, points[row]);
    }
  }

  Matrix top(k, std::vector<uint8_t>(k, 0));
  for (int row = 0; row < k; ++row) {
    top[row] = vandermonde[row];
  }
  Matrix top_inv = InvertMatrix(top);
  return MultiplyMatrices(vandermonde, top_inv);
}

Matrix BuildRecoveryMatrix(
    const Matrix& generator,
    const std::vector<int>& survivor_indices,
    const std::vector<int>& target_indices) {
  Matrix survivor_rows = SelectRows(generator, survivor_indices);
  Matrix inverse = InvertMatrix(survivor_rows);
  Matrix target_rows = SelectRows(generator, target_indices);
  return MultiplyMatrices(target_rows, inverse);
}

}  // namespace braid::core
